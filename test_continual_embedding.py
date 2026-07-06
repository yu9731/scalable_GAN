import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
import keras
import json
import h5py

def load_minmax_stats(stats_path):
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats

keras.config.enable_unsafe_deserialization()

data_length_lst = [30, 60, 90]
pred_hour = 48
variable_num = 4
e_dim = 64
building_sample = 181 - (pred_hour - 24) // 24

building_id_lst = [i for i in range(0, 54, 1)]
#building_id_lst = [i for i in range(0, 27, 1)]

for data_length in data_length_lst:
    pd_nrmse = pd.DataFrame(columns=["building", "method", "missing duration", "nrmse"])
    pd_mae = pd.DataFrame(columns=["building", "method", "missing duration", "mae"])

    def get_points(pred_hour, batch_size, variable_num, mask_len):
        mask = np.zeros((batch_size, pred_hour, variable_num, 1), dtype=np.float32)
        start_t = np.random.randint(0, pred_hour - mask_len + 1)
        mask[:, start_t:start_t + mask_len, 1, 0] = 1.0   # 1: target variable

        return mask, start_t, start_t + mask_len

    class NaiveEmbedding(tf.keras.Model):
        def __init__(
            self,
            N,
            num_primary,
            num_subtypes,
            num_energy,
            emb_building_dim=32,
            emb_meta_dim=16,
            hidden_dim=64,
            out_dim=64,
            dropout=0.1
        ):
            super().__init__()

            self.emb_building = tf.keras.layers.Embedding(
                input_dim=N,
                output_dim=emb_building_dim
            )

            self.emb_primary = tf.keras.layers.Embedding(
                input_dim=num_primary,
                output_dim=emb_meta_dim
            )

            self.emb_subtype = tf.keras.layers.Embedding(
                input_dim=num_subtypes,
                output_dim=emb_meta_dim
            )

            self.emb_energy = tf.keras.layers.Embedding(
                input_dim=num_energy,
                output_dim=emb_meta_dim
            )

            self.mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(out_dim)
            ])

        def call(self, node_feat_int, node_feat_float, training=False):
            b_id = node_feat_int[:, 0]
            p_id = node_feat_int[:, 1]
            s_id = node_feat_int[:, 2]
            e_id = node_feat_int[:, 3]

            b_emb = self.emb_building(b_id)
            p_emb = self.emb_primary(p_id)
            s_emb = self.emb_subtype(s_id)
            e_emb = self.emb_energy(e_id)

            x = tf.concat(
                [b_emb, p_emb, s_emb, e_emb, node_feat_float],
                axis=-1
            )

            E = self.mlp(x, training=training)

            return E


    def collect_h5_datasets(h5_path):
        datasets = []

        with h5py.File(h5_path, "r") as f:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    arr = np.array(obj)
                    datasets.append((name, arr))

            f.visititems(visitor)

        return datasets

    def infer_saved_building_embedding_N(h5_path, emb_building_dim=32):
        datasets = collect_h5_datasets(h5_path)

        candidates = []

        for name, arr in datasets:
            if arr.ndim == 2 and arr.shape[1] == emb_building_dim:
                candidates.append((name, arr.shape))

        if len(candidates) == 0:
            raise ValueError(
                "Cannot infer building embedding size from weights file. "
                "No 2D dataset with second dimension = emb_building_dim was found."
            )

        # building embedding is usually the only matrix with shape (N, 32)
        # Dense kernels usually have second dim 64, not 32.
        name, shape = candidates[0]
        print("Detected building embedding weight:", name, shape)

        return int(shape[0])


    def load_weights_robust(model, weights_path):
        try:
            model.load_weights(weights_path)
            print("✅ Naive embedding loaded with Keras load_weights().")
            return
        except Exception as e:
            print("⚠️ Keras load_weights() failed.")
            print("Reason:", str(e))
            print("Trying manual shape-based loading...")

        datasets = collect_h5_datasets(weights_path)

        used = set()
        assigned = 0

        for var in model.weights:
            var_shape = tuple(var.shape)

            matched_idx = None

            for idx, (name, arr) in enumerate(datasets):
                if idx in used:
                    continue

                if tuple(arr.shape) == var_shape:
                    matched_idx = idx
                    break

            if matched_idx is None:
                raise ValueError(
                    f"Cannot find matching weight for variable {var.name}, "
                    f"shape={var_shape}"
                )

            name, arr = datasets[matched_idx]
            var.assign(arr)

            used.add(matched_idx)
            assigned += 1

            print(f"Loaded {var.name} <= {name}, shape={arr.shape}")

        print(f"✅ Manual loading finished. Assigned {assigned} variables.")

    embed_weights_path = f"when2heat-gnn/mixed_energy/model/continual/continual_embed_{data_length}.weights.h5"
    Node_feat_new = np.load("when2heat-gnn/mixed_energy/continual/Node_feat_continual.npy").astype(np.float32)

    N_new = Node_feat_new.shape[0]

    saved_N = infer_saved_building_embedding_N(
        embed_weights_path,
        emb_building_dim=32
    )

    print("N_new:", N_new)
    print("saved_N from continual_embed.weights.h5:", saved_N)

    use_global_old_new_embedding = False
    N_old = 0

    if saved_N == N_new:
        Node_feat = Node_feat_new
        N = N_new
        use_global_old_new_embedding = False
        print("Using continual-only embedding table.")

    else:
        Node_feat_rep = np.load(
            "when2heat-gnn/mixed_energy/Node_feat.npy"
        ).astype(np.float32)

        N_old = Node_feat_rep.shape[0]

        if saved_N != N_old + N_new:
            raise ValueError(
                f"Embedding weight size mismatch: saved_N={saved_N}, "
                f"N_new={N_new}, N_old+N_new={N_old + N_new}. "
                "The testing embedding size does not match the saved continual_embed.weights.h5."
            )

        Node_feat = np.concatenate(
            [Node_feat_rep, Node_feat_new],
            axis=0
        )

        N = saved_N
        use_global_old_new_embedding = True

        print("Using old+new global embedding table.")
        print("N_old:", N_old)
        print("N_new:", N_new)
        print("N_total:", N)


    num_primary = 3
    num_subtypes = 5
    num_energy = 3

    sqm = Node_feat[:, 3:4]
    sqm_mean = sqm.mean(axis=0, keepdims=True)
    sqm_std = sqm.std(axis=0, keepdims=True) + 1e-8
    sqm_scaled = (sqm - sqm_mean) / sqm_std

    building_id_arr = np.arange(N, dtype=np.int32).reshape(N, 1)

    node_ids_int = np.hstack([
        building_id_arr,
        Node_feat[:, 0:3].astype(np.int32)
    ]).astype(np.int32)

    node_feat_float = sqm_scaled.astype(np.float32)

    Node_feat_int_tf = tf.constant(node_ids_int, dtype=tf.int32)
    Node_feat_float_tf = tf.constant(node_feat_float, dtype=tf.float32)


    # ============================================================
    # Build and load embedding
    # ============================================================

    naive_embed = NaiveEmbedding(
        N=N,
        num_primary=num_primary,
        num_subtypes=num_subtypes,
        num_energy=num_energy,
        emb_building_dim=32,
        emb_meta_dim=16,
        hidden_dim=64,
        out_dim=e_dim,
        dropout=0.1
    )

    # build once before loading weights
    _ = naive_embed(
        Node_feat_int_tf,
        Node_feat_float_tf,
        training=False
    )

    load_weights_robust(
        naive_embed,
        embed_weights_path
    )

    # Precompute node embeddings for all buildings
    E_all = naive_embed(
        Node_feat_int_tf,
        Node_feat_float_tf,
        training=False
    )   # (N, e_dim)

    #X_test = np.load(f"when2heat-gnn/mixed_energy/continual/mixed_test_continual_{data_length}.npy")
    #BID_test = np.load(f"when2heat-gnn/mixed_energy/continual/mixed_building_idx_continual_{data_length}_test.npy").astype(np.int32)

    X_test = np.load(f"when2heat-gnn/mixed_energy/mixed_test.npy")
    BID_test = np.load(f"when2heat-gnn/mixed_energy/mixed_building_idx_test.npy").astype(np.int32)

    '''
    def normalize_test_with_saved_minmax(
        X_test,
        building_idx_test,
        stats,
        eps=1e-8,
        clip=True
    ):
        X_test_norm = np.zeros_like(X_test, dtype=np.float32)
        unique_buildings = np.unique(building_idx_test)

        for b in unique_buildings:
            b_key = str(int(b))

            sample_mask = building_idx_test == b
            X_b = X_test[sample_mask]

            x_min = np.array(stats[b_key]["min"], dtype=np.float32)
            x_max = np.array(stats[b_key]["max"], dtype=np.float32)

            x_min = x_min.reshape(1, 1, -1, 1)
            x_max = x_max.reshape(1, 1, -1, 1)

            denom = x_max - x_min
            denom = np.where(denom < eps, eps, denom)

            X_b_norm = (X_b - x_min) / denom

            if clip:
                X_b_norm = np.clip(X_b_norm, 0.0, 1.0)

            X_test_norm[sample_mask] = X_b_norm

        return X_test_norm.astype(np.float32)


    stats = load_minmax_stats(
        f"when2heat-gnn/mixed_energy/continual/new_building_minmax_stats_{data_length}.json"
    )

    X_test = normalize_test_with_saved_minmax(
        X_test=X_test,
        building_idx_test=BID_test,
        stats=stats,
        clip=True
    )
    '''

    # ============================================================
    # Load generator
    # ============================================================

    generator = load_model(
        f"when2heat-gnn/mixed_energy/model/continual/generator_continual_embedded_{data_length}.h5",
        compile=False
    )
    print("✅ Generator loaded.")

    mask_len_lst = [6, 12, 18, 24, 30, 36, 42]

    for mask_len in mask_len_lst:
        np_err = np.zeros((90 * mask_len, len(building_id_lst)))
        for k, building_id in enumerate(building_id_lst):
            x_in = X_test[
                building_sample * building_id:
                building_sample * (building_id + 1)
            ]

            BID = BID_test[
                building_sample * building_id:
                building_sample * (building_id + 1)
            ]

            pred_chunks = []
            true_chunks = []

            max_data = np.load(f"when2heat-gnn/min_max/max_{k}.npy")
            min_data = np.load(f"when2heat-gnn/min_max/min_{k}.npy")

            batch_size = 1

            #for i in range(0, 90, batch_size):
            for i in range(90, 180, batch_size):
                xb = tf.convert_to_tensor(
                    x_in[i:i + batch_size],
                    dtype=tf.float32
                )

                b = int(BID[i])

                #if use_global_old_new_embedding:
                    #b = b + N_old

                e = E_all[b:b + 1]

                masks, start_idx, end_idx = get_points(
                    pred_hour,
                    batch_size,
                    variable_num,
                    mask_len
                )

                gen_input = xb * (1.0 - masks)

                yb = generator(
                    [gen_input, e],
                    training=False
                )   # (1, 48, 4, 1)

                pred_chunks.append(yb.numpy()[:, start_idx:end_idx, :, :])
                true_chunks.append(xb.numpy()[:, start_idx:end_idx, :, :])

            y_pred = np.concatenate(pred_chunks, axis=0)
            y_true = np.concatenate(true_chunks, axis=0)

            pred_load_norm = y_pred[:, :, 1, 0].reshape(-1)
            true_load_norm = y_true[:, :, 1, 0].reshape(-1)

            pred_load = pred_load_norm  #* (max_data - min_data) + min_data
            true_load = true_load_norm  #* (max_data - min_data) + min_data

            denom = np.max(true_load) - np.min(true_load)

            if denom < 1e-8:
                NRMSE = np.nan
                NMAE = np.nan
            else:
                NRMSE = np.sqrt(np.mean((pred_load - true_load) ** 2)) / denom
                NMAE = np.mean(np.abs(pred_load - true_load)) / denom

            print("====================================")
            print(f"Building {building_id} | Variant B (your trained GCN = building+type embeddings)")
            print(f"MAE   = {NMAE:.4f}")
            print(f"NRMSE = {NRMSE:.4f}")
            print("====================================")

            #plt.figure(figsize=(10, 4))
            #plt.plot(true_load[0:24 * 30], label="True Load")
            #plt.plot(pred_load[0:24 * 30], label="Simulated Load")
            #plt.legend()
            #plt.show()

            pd_nrmse.loc[k] = [building_id, "Continual", mask_len, NRMSE]
            pd_mae.loc[k] = [building_id, "Continual", mask_len, NMAE]

        #pd_nrmse.to_csv(f"result/continual/mixed_continual_{data_length}_{mask_len}.csv", index=False) #_warm_month
        #pd_mae.to_csv(f"result/continual/NMAE_mixed_continual_{data_length}_{mask_len}.csv", index=False)
        pd_nrmse.to_csv(f"result/continual/mixed_forgetting_{data_length}_{mask_len}_warm_month.csv", index=False)  # _warm_month
        pd_mae.to_csv(f"result/continual/NMAE_mixed_forgetting_{data_length}_{mask_len}_warm_month.csv", index=False)