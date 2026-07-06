import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
import keras


keras.config.enable_unsafe_deserialization()

pred_hour = 48
variable_num = 4
e_dim = 64
building_sample = 181 - (pred_hour - 24) // 24

building_id_lst = [36, 38, 51]   # 36, 38, 51
# [i for i in range(0, 54, 1)]
# building_id_min_max_lst = np.load("when2heat-gnn/mixed_energy/idx_usage_type.npy")[36:54]

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

        self.emb_building = tf.keras.layers.Embedding(input_dim=N, output_dim=emb_building_dim)
        self.emb_primary  = tf.keras.layers.Embedding(input_dim=num_primary, output_dim=emb_meta_dim)
        self.emb_subtype  = tf.keras.layers.Embedding(input_dim=num_subtypes, output_dim=emb_meta_dim)
        self.emb_energy   = tf.keras.layers.Embedding(input_dim=num_energy, output_dim=emb_meta_dim)

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
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

        X0 = tf.concat([b_emb, p_emb, s_emb, e_emb, node_feat_float], axis=-1)
        E = self.mlp(X0, training=training)
        return E

generator = load_model(
    "when2heat-gnn/mixed_energy/model/generator_aphere_random.h5",
    compile=False
)
# generator_aphere_random.h5
print("✅ Generator loaded.")

Node_feat = np.load("when2heat-gnn/mixed_energy/Node_feat.npy") #[36:54]
# continual/Node_feat_continual.npy
# mixed_energy/Node_feat.npy
N = Node_feat.shape[0]
# Node_feat

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

# -------------------------------------------------
# Load naive embedding model
# -------------------------------------------------
naive_embed = NaiveEmbedding(
    N=N,
    num_primary=num_primary,
    num_subtypes=num_subtypes,
    num_energy=num_energy,
    out_dim=e_dim
)

# build once before loading weights
_ = naive_embed(Node_feat_int_tf, Node_feat_float_tf, training=False)

naive_embed.load_weights(
    "when2heat-gnn/mixed_energy/model/naive_embed_random.weights.h5"
)
print("✅ Naive embedding loaded.")
# naive_embed_ori
# naive_embed_random

# Precompute node embeddings for all buildings
E_all = naive_embed(Node_feat_int_tf, Node_feat_float_tf, training=False)   # (N, e_dim)
#X_test = np.load("when2heat-gnn/mixed_energy/mixed_energy/mixed_test_continual_60.npy")#[building_sample*36:building_sample*54]
#BID_test = np.load("when2heat-gnn/mixed_energy/mixed_energy/mixed_building_idx_continual_60_test.npy")#[building_sample*36:building_sample*54]
X_test = np.load("when2heat-gnn/mixed_energy/mixed_energy/mixed_test.npy") #[building_sample*36:building_sample*54]
BID_test = np.load("when2heat-gnn/mixed_energy/mixed_energy/mixed_building_idx_test.npy") #[building_sample*36:building_sample*54]

# mask_len_lst = [6, 12, 18, 24, 30, 36, 42]
mask_len_lst = [12, 24, 36]

for mask_len in mask_len_lst:
    np_err = np.zeros((90 * mask_len, len(building_id_lst)))

    for k, building_id in enumerate(building_id_lst):
        x_in = X_test[building_sample * building_id:building_sample * (building_id + 1)]
        BID = BID_test[building_sample * building_id:building_sample * (building_id + 1)]

        pred_chunks = []
        true_chunks = []

        #max_data = np.load(f"when2heat-gnn/min_max/max_{building_id_min_max_lst[k]}.npy")
        #min_data = np.load(f"when2heat-gnn/min_max/min_{building_id_min_max_lst[k]}.npy")

        max_data = np.load(f"when2heat-gnn/min_max/max_{building_id}.npy")
        min_data = np.load(f"when2heat-gnn/min_max/min_{building_id}.npy")

        batch_size = 1
        # for i in range(0, 90, batch_size):
        # for i in range(90, 180, batch_size):

        for i in range(0, 90, 2):
        # for i in range(90, 180, 2):
            # xb = tf.convert_to_tensor(x_in[i:i + batch_size], dtype=tf.float32)   # (B,48,4,1)
            xb = tf.convert_to_tensor(x_in[i:i + 1], dtype=tf.float32)
            b = int(BID[i])  # building index used during training
            e = E_all[b:b+1]
            #e = E_all[(b-36):(b-35)]

            masks, start_idx, end_idx = get_points(pred_hour, xb.shape[0], variable_num, mask_len)
            gen_input = xb * (1.0 - masks)

            yb = generator([gen_input, e], training=False)

            # pred_chunks.append(yb.numpy()[:, start_idx:end_idx, :, :])
            # true_chunks.append(xb.numpy()[:, start_idx:end_idx, :, :])

            combined = xb.numpy().copy()
            combined[:, start_idx:end_idx, :, :] = yb.numpy()[:, start_idx:end_idx, :, :]

            pred_chunks.append(combined[:, :, :, :])
            true_chunks.append(xb.numpy()[:, :, :, :])

            #if len(pred_chunks) == 0:
                #pred_chunks.append(combined)
                #true_chunks.append(xb.numpy())
            #else:
                #pred_chunks.append(combined[:, 24:, :, :])
                #true_chunks.append(xb.numpy()[:, 24:, :, :])

        y_pred = np.concatenate(pred_chunks, axis=1)
        y_true = np.concatenate(true_chunks, axis=1)

        pred_load_norm = y_pred[:, :, 1, 0].reshape(-1)
        true_load_norm = y_true[:, :, 1, 0].reshape(-1)

        pred_load = pred_load_norm * (max_data - min_data) + min_data
        true_load = true_load_norm * (max_data - min_data) + min_data

        NMAE = np.mean(np.abs(pred_load - true_load)) / (np.max(pred_load) - np.min(true_load))
        NRMSE = np.sqrt(np.mean((pred_load_norm - true_load_norm) ** 2)) / (np.max(true_load_norm) - np.min(true_load_norm))

        print("====================================")
        print(f"Building {building_id}")
        print(f"MAE   = {NMAE:.4f}")
        print(f"NRMSE = {NRMSE:.4f}")
        print("====================================")

        #plt.figure(figsize=(10, 4))
        #plt.plot(true_load[0:24 * 12], label="True Load")
        #plt.plot(pred_load[0:24 * 12], label="Simulated Load")
        #plt.legend()
        #plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(true_load[24*30:24*60], label="True Load")
        plt.plot(pred_load[24*30:24*60], label="Simulated Load")
        plt.legend()
        plt.show()

        #pd_nrmse.loc[k] = [building_id, "Only office", mask_len, NRMSE]
        #pd_mae.loc[k] = [building_id, 'Only office', mask_len, NMAE]

        #pd_nrmse.loc[k] = [building_id, "Scratch", mask_len, NRMSE]
        #pd_mae.loc[k] = [building_id, 'Scratch', mask_len, NMAE]

        #np.save(f'visu_comparison/ori_{building_id}_{mask_len}.npy', true_load[24*30:24*60])
        #np.save(f'visu_comparison/rec_{building_id}_{mask_len}.npy', pred_load[24*30:24*60])

    #pd_nrmse.to_csv(f"result/mixed_energy/mixed_embedded_{mask_len}.csv", index=False)
    #pd_mae.to_csv(f'result/mixed_energy/NMAE_mixed_embedded_{mask_len}.csv', index=False)

    #pd_nrmse.to_csv(f"result/mixed_energy/mixed_60_ori_{mask_len}_warm_month.csv", index=False)
    #pd_mae.to_csv(f'result/mixed_energy/NMAE_mixed_60_ori_{mask_len}_warm_month.csv', index=False)