import os
import time
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print("Error enabling memory growth:", e)


# ============================================================
# Config
# ============================================================

pred_hour = 48
variable_num = 4

num_primary = 3
num_subtypes = 5
num_energy = 3

e_dim = 64
epochs = 200
data_length = 30

lambda_l1 = 100.0
lambda_adv = 1.0

lambda_new = 1.0
lambda_rep = 1.0

target_var_idx = 1
start_t = 24

batch_size_rep = 128
batch_size_new = 64

save_dir = "when2heat-gnn/mixed_energy/model/continual"
os.makedirs(save_dir, exist_ok=True)

norm_dir = "when2heat-gnn/mixed_energy/continual"
os.makedirs(norm_dir, exist_ok=True)

X_rep = np.load(
    "when2heat-gnn/mixed_energy/mixed_train.npy"
).astype(np.float32)

building_idx_rep = np.load(
    "when2heat-gnn/mixed_energy/mixed_building_idx_train.npy"
).astype(np.int32)

Node_feat_rep = np.load(
    "when2heat-gnn/mixed_energy/Node_feat.npy"
).astype(np.float32)

X_new = np.load(
    f"when2heat-gnn/mixed_energy/continual/mixed_train_continual_{data_length}.npy"
).astype(np.float32)

print(X_new.shape)

building_idx_new = np.load(
    f"when2heat-gnn/mixed_energy/continual/mixed_building_idx_continual_{data_length}_train.npy"
).astype(np.int32)

Node_feat_new = np.load(
    "when2heat-gnn/mixed_energy/continual/Node_feat_continual.npy"
).astype(np.float32)


# ============================================================
# Normalization
# ============================================================

def normalize_per_building_minmax(
    X,
    building_idx,
    save_path=None,
    eps=1e-8,
    lower_q=None,
    upper_q=None
):
    X_norm = np.zeros_like(X, dtype=np.float32)
    stats = {}

    unique_buildings = np.unique(building_idx)

    for b in unique_buildings:
        sample_mask = building_idx == b
        X_b = X[sample_mask]

        if lower_q is None or upper_q is None:
            x_min = np.min(X_b, axis=(0, 1), keepdims=True)
            x_max = np.max(X_b, axis=(0, 1), keepdims=True)
        else:
            x_min = np.percentile(
                X_b,
                lower_q,
                axis=(0, 1),
                keepdims=True
            )
            x_max = np.percentile(
                X_b,
                upper_q,
                axis=(0, 1),
                keepdims=True
            )

        denom = x_max - x_min
        denom = np.where(denom < eps, eps, denom)

        X_b_norm = (X_b - x_min) / denom
        X_b_norm = np.clip(X_b_norm, 0.0, 1.0)

        X_norm[sample_mask] = X_b_norm

        stats[int(b)] = {
            "min": x_min.squeeze().tolist(),
            "max": x_max.squeeze().tolist()
        }

    X_norm[np.isnan(X_norm)] = 0.0

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=4)

    return X_norm.astype(np.float32), stats

X_new, stats_new = normalize_per_building_minmax(
    X=X_new,
    building_idx=building_idx_new,
    save_path=os.path.join(norm_dir, f"new_building_minmax_stats_{data_length}.json"),
    lower_q=1,
    upper_q=99
)

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


def prepare_node_features(
    Node_feat,
    id_offset=0,
    sqm_mean=None,
    sqm_std=None
):
    N = Node_feat.shape[0]

    sqm = Node_feat[:, 3:4].astype(np.float32)

    if sqm_mean is None:
        sqm_mean = sqm.mean(axis=0, keepdims=True)

    if sqm_std is None:
        sqm_std = sqm.std(axis=0, keepdims=True) + 1e-8

    sqm_scaled = (sqm - sqm_mean) / sqm_std

    building_id = (
        np.arange(N, dtype=np.int32).reshape(N, 1) + id_offset
    )

    node_ids_int = np.hstack([
        building_id,
        Node_feat[:, 0:3].astype(np.int32)
    ]).astype(np.int32)

    node_feat_float = sqm_scaled.astype(np.float32)

    Node_feat_int_tf = tf.constant(node_ids_int, dtype=tf.int32)
    Node_feat_float_tf = tf.constant(node_feat_float, dtype=tf.float32)

    return Node_feat_float_tf, Node_feat_int_tf, sqm_mean, sqm_std

N_old = Node_feat_rep.shape[0]
N_new = Node_feat_new.shape[0]
N_total = N_old + N_new

print("N_old:", N_old)
print("N_new:", N_new)
print("N_total:", N_total)

# Old building ids: 0 ... N_old - 1
Node_feat_float_tf_rep, Node_feat_int_tf_rep, sqm_mean_rep, sqm_std_rep = prepare_node_features(
    Node_feat_rep,
    id_offset=0
)

# New building ids: N_old ... N_old + N_new - 1
Node_feat_float_tf_new, Node_feat_int_tf_new, _, _ = prepare_node_features(
    Node_feat_new,
    id_offset=N_old,
    sqm_mean=sqm_mean_rep,
    sqm_std=sqm_std_rep
)

Node_feat_int_tf_all = tf.concat(
    [Node_feat_int_tf_rep, Node_feat_int_tf_new],
    axis=0
)

Node_feat_float_tf_all = tf.concat(
    [Node_feat_float_tf_rep, Node_feat_float_tf_new],
    axis=0
)

building_idx_rep = building_idx_rep.astype(np.int32)
building_idx_new = building_idx_new.astype(np.int32) + N_old

naive_embed_new = NaiveEmbedding(
    N=N_total,
    num_primary=num_primary,
    num_subtypes=num_subtypes,
    num_energy=num_energy,
    emb_building_dim=32,
    emb_meta_dim=16,
    hidden_dim=64,
    out_dim=e_dim,
    dropout=0.1
)

# build variables
_ = naive_embed_new(
    Node_feat_int_tf_all,
    Node_feat_float_tf_all,
    training=True
)


# ============================================================
# Load generator
# ============================================================

generator = load_model(
    "when2heat-gnn/mixed_energy/model/generator_aphere_random.h5",
    compile=False
)

def freeze_batchnorm(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

        if hasattr(layer, "layers"):
            freeze_batchnorm(layer)

freeze_batchnorm(generator)

def downsample(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    seq = tf.keras.Sequential()

    seq.add(tf.keras.layers.Conv2D(
        filters,
        size,
        strides=strides,
        padding="same",
        kernel_initializer=initializer,
        use_bias=False
    ))

    if apply_batchnorm:
        seq.add(tf.keras.layers.BatchNormalization())

    seq.add(tf.keras.layers.LeakyReLU())

    return seq


def Discriminator(pred_hour, variable_num, e_dim=64):
    inp = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1])
    tar = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1])
    e_in = tf.keras.layers.Input(shape=(e_dim,))

    e_proj = tf.keras.layers.Dense(
        pred_hour * variable_num,
        activation="relu"
    )(e_in)

    e_map = tf.keras.layers.Reshape(
        (pred_hour, variable_num, 1)
    )(e_proj)

    x = tf.keras.layers.Concatenate(axis=-1)([
        inp,
        tar,
        e_map
    ])

    d1 = downsample(32, 3, (2, 1), False)(x)
    d2 = downsample(64, 3, (2, 1))(d1)
    d3 = downsample(128, 3, (2, 1))(d2)

    z1 = tf.keras.layers.ZeroPadding2D()(d3)

    conv = tf.keras.layers.Conv2D(
        128,
        3,
        strides=1,
        use_bias=False
    )(z1)

    bn = tf.keras.layers.BatchNormalization()(conv)
    act = tf.keras.layers.LeakyReLU()(bn)

    z2 = tf.keras.layers.ZeroPadding2D()(act)

    last = tf.keras.layers.Conv2D(
        1,
        3,
        strides=1,
        activation=None
    )(z2)

    return tf.keras.Model(
        inputs=[inp, tar, e_in],
        outputs=last,
        name="Discriminator_GNN"
    )


discriminator = Discriminator(pred_hour, variable_num, e_dim)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def get_points_tf(batch_size, pred_hour, variable_num, target_var_idx=1):
    mask_len = tf.random.uniform(shape=(),minval=6,maxval=43,dtype=tf.int32)
    start_t = tf.random.uniform(shape=(),minval=0,maxval=pred_hour - mask_len + 1,dtype=tf.int32)

    time_idx = tf.range(pred_hour, dtype=tf.int32)  # (pred_hour,)
    time_mask = tf.logical_and(time_idx >= start_t,time_idx < start_t + mask_len)

    time_mask = tf.cast(time_mask, tf.float32)
    time_mask = tf.reshape(time_mask, (pred_hour, 1, 1))  # (pred_hour, 1, 1)

    target_col = tf.one_hot(target_var_idx,depth=variable_num,dtype=tf.float32)

    target_col = tf.reshape(target_col, (1, variable_num, 1))

    mask = time_mask * target_col

    mask = tf.expand_dims(mask, axis=0)
    mask = tf.tile(mask, [batch_size, 1, 1, 1])

    return mask

def masked_l1_loss(y_true, y_pred):  #, mask, eps=1e-8):
    abs_err = tf.abs(y_true - y_pred) #* mask
    loss = tf.reduce_sum(abs_err) #/ (tf.reduce_sum(mask) + eps)
    return loss

def generator_loss(
    disc_generated_output,
    gen_output,
    target,
    lambda_l1=100.0,
    lambda_adv=1.0
):
    adv_loss = loss_object(
        tf.ones_like(disc_generated_output),
        disc_generated_output
    )

    l1_loss = masked_l1_loss(
        target,
        gen_output
    )

    total_loss = lambda_adv * adv_loss + lambda_l1 * l1_loss

    return total_loss, adv_loss, l1_loss

def discriminator_loss(
    disc_real_output,
    disc_generated_output
):
    real_loss = loss_object(
        tf.ones_like(disc_real_output),
        disc_real_output
    )

    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output),
        disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def make_dataset(X, idx, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), idx.astype(np.int32))
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(X),
            reshuffle_each_iteration=True
        )

    return ds.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)


new_iter = iter(
    make_dataset(
        X_new,
        building_idx_new,
        batch_size_new,
        shuffle=True
    )
)

rep_iter = iter(
    make_dataset(
        X_rep,
        building_idx_rep,
        batch_size_rep,
        shuffle=True
    )
)

steps_per_epoch = X_rep.shape[0] // batch_size_rep

print("steps_per_epoch:", steps_per_epoch)

optimizer_g = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
optimizer_d = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
optimizer_embed = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

@tf.function
def train_step(x_new, b_new, x_rep, b_rep):
    new_bs = tf.shape(x_new)[0]
    rep_bs = tf.shape(x_rep)[0]

    masks_new = get_points_tf(
        new_bs,
        pred_hour,
        variable_num,
        target_var_idx=target_var_idx,
    )

    masks_rep = get_points_tf(
        rep_bs,
        pred_hour,
        variable_num,
        target_var_idx=target_var_idx,
    )

    gen_input_new = x_new * (1.0 - masks_new)
    gen_input_rep = x_rep * (1.0 - masks_rep)

    with tf.GradientTape(persistent=True) as tape:
        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------
        E_all = naive_embed_new(
            Node_feat_int_tf_all,
            Node_feat_float_tf_all,
            training=True
        )

        e_batch_new = tf.gather(
            E_all,
            tf.cast(b_new, tf.int32)
        )

        e_batch_rep = tf.gather(
            E_all,
            tf.cast(b_rep, tf.int32)
        )

        # ----------------------------------------------------
        # Generator outputs
        # ----------------------------------------------------
        gen_output_new = generator(
            [gen_input_new, e_batch_new],
            training=True
        )

        gen_output_rep = generator(
            [gen_input_rep, e_batch_rep],
            training=True
        )

        disc_generated_new_output_for_g = discriminator(
            [gen_input_new, gen_output_new, e_batch_new],
            training=True
        )

        disc_generated_rep_output_for_g = discriminator(
            [gen_input_rep, gen_output_rep, e_batch_rep],
            training=True
        )

        g_loss_new, g_adv_new, g_l1_new = generator_loss(
            disc_generated_new_output_for_g,
            gen_output_new,
            x_new,
            lambda_l1=lambda_l1,
            lambda_adv=lambda_adv
        )

        g_loss_rep, g_adv_rep, g_l1_rep = generator_loss(
            disc_generated_rep_output_for_g,
            gen_output_rep,
            x_rep,
            lambda_l1=lambda_l1,
            lambda_adv=lambda_adv
        )

        total_g_loss = (
            lambda_new * g_loss_new
            + lambda_rep * g_loss_rep
        )

        disc_real_new_output = discriminator(
            [gen_input_new, x_new, e_batch_new],
            training=True
        )

        disc_generated_new_output_for_d = discriminator(
            [
                gen_input_new,
                tf.stop_gradient(gen_output_new),
                e_batch_new
            ],
            training=True
        )

        disc_real_rep_output = discriminator(
            [gen_input_rep, x_rep, e_batch_rep],
            training=True
        )

        disc_generated_rep_output_for_d = discriminator(
            [
                gen_input_rep,
                tf.stop_gradient(gen_output_rep),
                e_batch_rep
            ],
            training=True
        )

        d_loss_new = discriminator_loss(
            disc_real_new_output,
            disc_generated_new_output_for_d
        )

        d_loss_rep = discriminator_loss(
            disc_real_rep_output,
            disc_generated_rep_output_for_d
        )

        total_d_loss = d_loss_new + d_loss_rep

    grads_g = tape.gradient(
        total_g_loss,
        generator.trainable_variables
    )

    grads_embed = tape.gradient(
        total_g_loss,
        naive_embed_new.trainable_variables
    )

    grads_d = tape.gradient(
        total_d_loss,
        discriminator.trainable_variables
    )

    optimizer_g.apply_gradients(
        [
            (g, v)
            for g, v in zip(grads_g, generator.trainable_variables)
            if g is not None
        ]
    )

    optimizer_embed.apply_gradients(
        [
            (g, v)
            for g, v in zip(grads_embed, naive_embed_new.trainable_variables)
            if g is not None
        ]
    )

    optimizer_d.apply_gradients(
        [
            (g, v)
            for g, v in zip(grads_d, discriminator.trainable_variables)
            if g is not None
        ]
    )

    del tape

    return (
        total_g_loss,
        total_d_loss,
        g_loss_new,
        g_loss_rep,
        g_adv_new,
        g_adv_rep,
        g_l1_new,
        g_l1_rep
    )


# ============================================================
# Training loop
# ============================================================

history = {
    "G": [],
    "D": [],
    "G_new": [],
    "G_rep": [],
    "G_adv_new": [],
    "G_adv_rep": [],
    "G_l1_new": [],
    "G_l1_rep": []
}

for epoch in range(epochs + 1):
    start = time.time()

    g_losses = []
    d_losses = []
    g_new_losses = []
    g_rep_losses = []
    g_adv_new_losses = []
    g_adv_rep_losses = []
    g_l1_new_losses = []
    g_l1_rep_losses = []

    for step in range(steps_per_epoch):
        x_new_batch, b_new_batch = next(new_iter)
        x_rep_batch, b_rep_batch = next(rep_iter)

        x_new_batch = tf.cast(x_new_batch, tf.float32)
        b_new_batch = tf.cast(b_new_batch, tf.int32)

        x_rep_batch = tf.cast(x_rep_batch, tf.float32)
        b_rep_batch = tf.cast(b_rep_batch, tf.int32)

        (
            total_g_loss,
            total_d_loss,
            g_loss_new,
            g_loss_rep,
            g_adv_new,
            g_adv_rep,
            g_l1_new,
            g_l1_rep
        ) = train_step(
            x_new_batch,
            b_new_batch,
            x_rep_batch,
            b_rep_batch
        )

        g_losses.append(float(total_g_loss.numpy()))
        d_losses.append(float(total_d_loss.numpy()))
        g_new_losses.append(float(g_loss_new.numpy()))
        g_rep_losses.append(float(g_loss_rep.numpy()))
        g_adv_new_losses.append(float(g_adv_new.numpy()))
        g_adv_rep_losses.append(float(g_adv_rep.numpy()))
        g_l1_new_losses.append(float(g_l1_new.numpy()))
        g_l1_rep_losses.append(float(g_l1_rep.numpy()))

    mean_g = np.mean(g_losses)
    mean_d = np.mean(d_losses)
    mean_g_new = np.mean(g_new_losses)
    mean_g_rep = np.mean(g_rep_losses)
    mean_g_adv_new = np.mean(g_adv_new_losses)
    mean_g_adv_rep = np.mean(g_adv_rep_losses)
    mean_g_l1_new = np.mean(g_l1_new_losses)
    mean_g_l1_rep = np.mean(g_l1_rep_losses)

    history["G"].append(mean_g)
    history["D"].append(mean_d)
    history["G_new"].append(mean_g_new)
    history["G_rep"].append(mean_g_rep)
    history["G_adv_new"].append(mean_g_adv_new)
    history["G_adv_rep"].append(mean_g_adv_rep)
    history["G_l1_new"].append(mean_g_l1_new)
    history["G_l1_rep"].append(mean_g_l1_rep)

    print(
        f"Epoch {epoch:03d} | "
        f"G={mean_g:.6f} | "
        f"D={mean_d:.6f} | "
        f"G_new={mean_g_new:.6f} | "
        f"G_rep={mean_g_rep:.6f} | "
        f"adv_new={mean_g_adv_new:.6f} | "
        f"adv_rep={mean_g_adv_rep:.6f} | "
        f"l1_new={mean_g_l1_new:.6f} | "
        f"l1_rep={mean_g_l1_rep:.6f} | "
        f"time={time.time() - start:.1f}s"
    )

    if epoch % 10 == 0:
        generator.save(
            os.path.join(save_dir, f"generator_continual_embedded_{data_length}.h5")
        )

        discriminator.save(
            os.path.join(save_dir, f"discriminator_continual_embedded_{data_length}.h5")
        )

        naive_embed_new.save_weights(
            os.path.join(save_dir, f"continual_embed_{data_length}.weights.h5")
        )

        with open(
            os.path.join(save_dir, f"training_history_{data_length}.json"),
            "w"
        ) as f:
            json.dump(history, f, indent=4)

        print(f"Saved checkpoint at epoch {epoch}")

generator.save(
    os.path.join(save_dir, f"generator_continual_embedded_{data_length}.h5")
)

discriminator.save(
    os.path.join(save_dir, f"discriminator_continual_embedded_{data_length}.h5")
)

naive_embed_new.save_weights(
    os.path.join(save_dir, f"continual_embed_{data_length}.weights.h5")
)

with open(
    os.path.join(save_dir, f"training_history_{data_length}.json"),
    "w"
) as f:
    json.dump(history, f, indent=4)

print("Training finished.")