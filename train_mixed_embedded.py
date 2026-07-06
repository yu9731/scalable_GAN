import os
import time
import numpy as np
import tensorflow as tf
import time
import psutil

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

pred_hour = 48
variable_num = 4

num_primary = 3
num_subtypes = 5
num_energy = 3

e_dim = 64
batch_size = 128 # 128
epochs = 200

lambda_l1 = 100.0

save_dir = "when2heat-gnn/mixed_energy/model"
os.makedirs(save_dir, exist_ok=True)

X_train = np.load("when2heat-gnn/mixed_energy/mixed_train.npy").astype(np.float32) #[334*36:334*54]
building_idx = np.load("when2heat-gnn/mixed_energy/mixed_building_idx_train.npy").astype(np.int32) #[334*36:334*54]
Node_feat = np.load("when2heat-gnn/mixed_energy/Node_feat.npy").astype(np.float32) #[36:54]

# type_lst = ['education', 'office', 'lodging']
#type_lst = ['electricity', 'steam', 'chilledwater']

#for e, type in enumerate(type_lst):
    # X_train = np.load("when2heat-gnn/mixed_energy/mixed_train_usage_type.npy").astype(np.float32)[334*e*18:334*(e+1)*18]
    # building_idx = np.load("when2heat-gnn/mixed_energy/mixed_building_idx_train_usage_type.npy").astype(np.int32)[334*e*18:334*(e+1)*18]
    # Node_feat = np.load("when2heat-gnn/mixed_energy/Node_feat_usage_type.npy").astype(np.float32)[e*18:(e+1)*18]

    # X_train = np.load("when2heat-gnn/mixed_energy/mixed_train.npy").astype(np.float32)[334*e*18:334*(e+1)*18]
    # building_idx = np.load("when2heat-gnn/mixed_energy/mixed_building_idx_train.npy").astype(np.int32)[334*e*18:334*(e+1)*18]
    # Node_feat = np.load("when2heat-gnn/mixed_energy/Node_feat.npy").astype(np.float32)[e*18:(e+1)*18]

#X_train = np.load("when2heat-gnn/mixed_energy/mixed_energy/mixed_train_continual_30.npy").astype(np.float32)#[334*36:334*54]
#building_idx = np.load("when2heat-gnn/mixed_energy/mixed_energy/mixed_building_idx_continual_30_train.npy").astype(np.int32)#[334*36:334*54]
#Node_feat = np.load("when2heat-gnn/mixed_energy/continual/Node_feat_continual.npy").astype(np.float32)#[36:54]

N = Node_feat.shape[0]

print("X_train shape:", X_train.shape)
print("building_idx shape:", building_idx.shape)
print("Node_feat shape:", Node_feat.shape)

sqm = Node_feat[:, 3:4]
sqm_mean = sqm.mean(axis=0, keepdims=True)
sqm_std = sqm.std(axis=0, keepdims=True) + 1e-8
sqm_scaled = (sqm - sqm_mean) / sqm_std

building_id = np.arange(N, dtype=np.int32).reshape(N, 1)
node_ids_int = np.hstack([
    building_id,
    Node_feat[:, 0:3].astype(np.int32)   # primary_id, subtype_id, energy_id
]).astype(np.int32)                      # shape (N,4)

node_feat_float = sqm_scaled.astype(np.float32)  # shape (N,1)

Node_feat_int_tf = tf.constant(node_ids_int, dtype=tf.int32)
Node_feat_float_tf = tf.constant(node_feat_float, dtype=tf.float32)

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

def upsample(filters, size, strides, apply_batchnorm=True, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    seq = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2DTranspose(
        filters, size, strides=strides, padding='same',
        kernel_initializer=initializer, use_bias=False
    ))
    if apply_batchnorm:
        seq.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        seq.add(tf.keras.layers.Dropout(0.5))
    seq.add(tf.keras.layers.LeakyReLU())
    return seq

def downsample(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    seq = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2D(
        filters, size, strides=strides, padding='same',
        kernel_initializer=initializer, use_bias=False
    ))
    if apply_batchnorm:
        seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.LeakyReLU())
    return seq

def dilated(filters, size, dilation_rate):
    initializer = tf.random_normal_initializer(0., 0.02)
    seq = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2D(
        filters, size, strides=1, padding='same',
        dilation_rate=dilation_rate,
        kernel_initializer=initializer, use_bias=False
    ))
    seq.add(tf.keras.layers.LeakyReLU())
    return seq

def Generator(pred_hour, variable_num, e_dim=64):
    inputs = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1])
    e_in   = tf.keras.layers.Input(shape=(e_dim,))

    # stronger conditioning
    e_proj = tf.keras.layers.Dense(pred_hour * variable_num, activation='relu')(e_in)
    e_map = tf.keras.layers.Reshape((pred_hour, variable_num, 1))(e_proj)

    x = tf.keras.layers.Concatenate(axis=-1)([inputs, e_map])

    # time-aware encoder
    x1 = downsample(64, 3, (2,1), apply_batchnorm=False)(x)   # 48 -> 24
    x2 = downsample(128, 3, (2,1))(x1)                        # 24 -> 12

    # bottleneck
    b = dilated(256, 3, (1,1))(x2)
    b = dilated(256, 3, (2,1))(b)
    b = dilated(256, 3, (4,1))(b)
    b = dilated(256, 3, (8,1))(b)

    # decoder
    u1 = upsample(128, 3, (1,1))(b)                           # 12 -> 24
    u1 = tf.keras.layers.Concatenate()([u1, x2])

    u2 = upsample(64, 3, (2,1))(u1)                           # 24 -> 48
    u2 = tf.keras.layers.Concatenate()([u2, x1])

    out = tf.keras.layers.Conv2DTranspose(1, 3, (2,1), padding='same',activation='softplus')(u2)

    return tf.keras.Model(inputs=[inputs, e_in], outputs=out, name="Generator_GNN")

def Discriminator(pred_hour, variable_num, e_dim=64):
    inp = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1])
    tar = tf.keras.layers.Input(shape=[pred_hour, variable_num, 1])
    e_in = tf.keras.layers.Input(shape=(e_dim,))

    e_proj = tf.keras.layers.Dense(pred_hour * variable_num, activation='relu')(e_in)
    e_map = tf.keras.layers.Reshape((pred_hour, variable_num, 1))(e_proj)

    x = tf.keras.layers.Concatenate(axis=-1)([inp, tar, e_map])

    d1 = downsample(32, 3, (2,1), False)(x)
    d2 = downsample(64, 3, (2,1))(d1)
    d3 = downsample(128, 3, (2,1))(d2)

    z1 = tf.keras.layers.ZeroPadding2D()(d3)
    conv = tf.keras.layers.Conv2D(128, 3, strides=1, use_bias=False)(z1)
    bn = tf.keras.layers.BatchNormalization()(conv)
    act = tf.keras.layers.LeakyReLU()(bn)

    z2 = tf.keras.layers.ZeroPadding2D()(act)
    last = tf.keras.layers.Conv2D(1, 3, strides=1, activation=None)(z2)

    return tf.keras.Model(inputs=[inp, tar, e_in], outputs=last, name="Discriminator_GNN")

generator = Generator(pred_hour, variable_num, e_dim)
discriminator = Discriminator(pred_hour, variable_num, e_dim)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, lambda_l1=100.0):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # only penalize masked region
    abs_err = tf.abs((target - gen_output))
    l1_loss = tf.reduce_sum(abs_err)

    total_loss = gan_loss + lambda_l1 * l1_loss
    return total_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

def get_points_tf(batch_size, pred_hour, variable_num, target_var_idx=1):
    # Random mask length: [6, 42]
    mask_len = tf.random.uniform(
        shape=(),
        minval=6,
        maxval=43,   # upper bound is exclusive
        dtype=tf.int32
    )

    # Random start point: [0, pred_hour - mask_len]
    start_t = tf.random.uniform(
        shape=(),
        minval=0,
        maxval=pred_hour - mask_len + 1,
        dtype=tf.int32
    )

    time_idx = tf.range(pred_hour, dtype=tf.int32)  # (pred_hour,)

    # True only for the randomly selected masked interval
    time_mask = tf.logical_and(
        time_idx >= start_t,
        time_idx < start_t + mask_len
    )

    time_mask = tf.cast(time_mask, tf.float32)
    time_mask = tf.reshape(time_mask, (pred_hour, 1, 1))  # (pred_hour, 1, 1)

    target_col = tf.one_hot(
        target_var_idx,
        depth=variable_num,
        dtype=tf.float32
    )

    target_col = tf.reshape(target_col, (1, variable_num, 1))  # (1, variable_num, 1)

    mask = time_mask * target_col  # (pred_hour, variable_num, 1)

    mask = tf.expand_dims(mask, axis=0)  # (1, pred_hour, variable_num, 1)
    mask = tf.tile(mask, [batch_size, 1, 1, 1])

    return mask

steps_per_epoch = X_train.shape[0] // batch_size

def make_dataset(X, building_idx):
    ds = tf.data.Dataset.from_tensor_slices((X, building_idx))
    ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

dataset = make_dataset(X_train, building_idx)

optimizer_g = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
optimizer_d = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
optimizer_embed = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

@tf.function
def train_step(x_batch, b_idx):
    current_bs = tf.shape(x_batch)[0]
    # masks = get_points_tf(current_bs, 48, variable_num, target_var_idx=1, start_t=24)
    masks = get_points_tf(current_bs, 48, variable_num, target_var_idx=1)

    with tf.GradientTape(persistent=True) as tape:
        E_all = naive_embed(Node_feat_int_tf, Node_feat_float_tf, training=True)
        naive_embed.summary()

        e_batch = tf.gather(E_all, tf.cast(b_idx, tf.int32))                          # (bs, e_dim)

        # masked input
        gen_input = x_batch * (1.0 - masks)
        # generator
        gen_output = generator([gen_input, e_batch], training=True)
        # discriminator
        disc_real_output = discriminator([gen_input, x_batch, e_batch], training=True)
        disc_generated_output = discriminator([gen_input, gen_output, e_batch], training=True)

        # losses
        #total_g_loss, gan_loss, masked_l1_loss = generator_loss(
            #disc_generated_output, gen_output, x_batch, masks, lambda_l1=lambda_l1
        #)
        total_g_loss, gan_loss, masked_l1_loss = generator_loss(
            disc_generated_output, gen_output, x_batch, lambda_l1=lambda_l1
        )
        total_d_loss = discriminator_loss(disc_real_output, disc_generated_output)

    grads_g = tape.gradient(total_g_loss, generator.trainable_variables)
    grads_d = tape.gradient(total_d_loss, discriminator.trainable_variables)
    grads_embed = tape.gradient(total_g_loss, naive_embed.trainable_variables)

    # gradient clipping
    grads_g = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads_g]
    grads_d = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads_d]
    grads_embed = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads_embed]

    optimizer_g.apply_gradients(zip(grads_g, generator.trainable_variables))
    optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    optimizer_embed.apply_gradients(zip(grads_embed, naive_embed.trainable_variables))

    return total_g_loss, total_d_loss, gan_loss, masked_l1_loss


start_time = time.time()

for epoch in range(epochs + 1):
    start = time.time()

    g_losses = []
    d_losses = []
    gan_losses = []
    recon_losses = []

    for step, (batch, batch_idx) in enumerate(dataset.take(steps_per_epoch)):
        batch = tf.cast(batch, tf.float32)
        batch_idx = tf.cast(batch_idx, tf.int32)

        g_loss, d_loss, gan_loss, recon_loss = train_step(batch, batch_idx)

        g_losses.append(float(g_loss.numpy()))
        d_losses.append(float(d_loss.numpy()))
        gan_losses.append(float(gan_loss.numpy()))
        recon_losses.append(float(recon_loss.numpy()))

    mean_g = np.mean(g_losses)
    mean_d = np.mean(d_losses)
    mean_gan = np.mean(gan_losses)
    mean_recon = np.mean(recon_losses)

    print(
        f"Epoch {epoch:03d} | "
        f"G={mean_g:.4f} | D={mean_d:.4f} | "
        f"GAN={mean_gan:.4f} | MaskL1={mean_recon:.4f} | "
        f"time={time.time()-start:.1f}s"
    )

    #if epoch % 10 == 0:
        #generator.save(os.path.join(save_dir, f"generator_aphere_random_{type}.h5"))
        # discriminator.save(os.path.join(save_dir, "discriminator_aphere_ori_30.h5"))
        #naive_embed.save_weights(os.path.join(save_dir, f"naive_embed_random_{type}.weights.h5"))

process = psutil.Process(os.getpid())
memory_before = process.memory_info().rss / 1024**2  # MB

end_time = time.time()
memory_after = process.memory_info().rss / 1024**2

training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")
print(f"Memory increase: {memory_after:.2f} MB")