import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.initializers import HeNormal
import numpy as np
from tensorflow.python.training.moving_averages import ExponentialMovingAverage
from tensorflow.keras.mixed_precision import LossScaleOptimizer

# @register_keras_serializable()
# class SpectralNormalization(layers.Wrapper):
#     def __init__(self, layer, power_iterations=3, **kwargs):
#         super().__init__(layer, **kwargs)
#         self.power_iterations = power_iterations
#
#     def build(self, input_shape):
#         super().build(input_shape)
#         if not self.built:
#             self.layer.build(input_shape)  # 确保被包装层已构建
#             self.w = self.layer.kernel
#             self.w_shape = self.w.shape.as_list()
#             self.u = self.add_weight(
#                 shape=(1, self.w_shape[-1]),
#                 initializer="random_normal",
#                 trainable=False,
#                 dtype=tf.float32
#             )
#             self.built = True
#
#     def call(self, inputs):
#         if not self.built:
#             self.build(inputs.shape)  # 防御性调用
#         if self.trainable:
#             w = tf.reshape(self.w, [-1, self.w_shape[-1]])
#             u_hat = self.u
#             for _ in range(self.power_iterations):
#                 v_hat = tf.math.l2_normalize(tf.matmul(u_hat, w, transpose_b=True))
#                 u_hat = tf.math.l2_normalize(tf.matmul(v_hat, w))
#             sigma = tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True)
#             sigma = tf.maximum(sigma, 1e-12)
#             self.layer.kernel.assign(self.w / sigma)
#         return self.layer(inputs)
#
#     def compute_output_spec(self, inputs):
#         return self.layer.compute_output_spec(inputs)  # 显式定义输出形状
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({"power_iterations": self.power_iterations})
#         return config

class SpectralNormalization(layers.Wrapper):
    """
    谱归一化层（Spectral Normalization）
    通过约束权重矩阵的谱范数稳定训练
    """
    def __init__(self, layer, power_iterations=5, **kwargs):
        super().__init__(layer, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        if not self.built:
            self.layer.build(input_shape)
            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_weight(
                shape=(1, self.w_shape[-1]),
                initializer="random_normal",
                trainable=False,
                dtype=tf.float32
            )
            self.built = True

    def call(self, inputs):
        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u_hat = self.u
        for _ in range(self.power_iterations):
            v_hat = tf.math.l2_normalize(tf.matmul(u_hat, w, transpose_b=True))
            u_hat = tf.math.l2_normalize(tf.matmul(v_hat, w))
        sigma = tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True)
        self.layer.kernel.assign(self.w / sigma)
        return self.layer(inputs)

# def perceptual_loss(y_true, y_pred):
#     vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
#     true_features = vgg(y_true)
#     pred_features = vgg(y_pred)
#     return tf.reduce_mean(tf.square(true_features - pred_features))

def perceptual_loss(y_true, y_pred):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # 提取多层级特征（如 block1, block3, block5）
    layer_names = ['block1_conv1', 'block3_conv3', 'block5_conv3']
    outputs = [vgg.get_layer(name).output for name in layer_names]
    feature_model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    true_features = feature_model(y_true)
    pred_features = feature_model(y_pred)

    loss = 0
    for t, p in zip(true_features, pred_features):
        loss += tf.reduce_mean(tf.square(t - p))
    return loss / len(layer_names)  # 平均多尺度损失

def color_consistency_loss(y_true, y_pred):
    # 计算 RGB 通道均值的差异
    true_mean = tf.reduce_mean(y_true, axis=[1, 2])
    pred_mean = tf.reduce_mean(y_pred, axis=[1, 2])
    return tf.reduce_mean(tf.abs(true_mean - pred_mean))

def smooth_l1_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))

def se_block(x, ratio=8):
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(channels//ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = layers.LayerNormalization()(se)  # 新增层归一化
    se = layers.Dense(channels, activation='sigmoid')(se)
    return layers.Multiply()([x, se])

class HDR_GAN:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.image_size = config['image_size']
        self.c_dim = config['c_dim']  # 通道数，通常为3（RGB）
        self.input_photo_num = config['input_photo_num']
        self.gf_dim = 64  # 生成器的卷积通道数
        self.df_dim = 64  # 判别器的卷积通道数
        self.num_res_blocks = 9  # 残差块的数量
        self.checkpoint_dir = config['checkpoint_dir']  # 模型保存路径
        self.model_name = "HDR_GAN"
        self.attention_process = tf.Variable(
            initial_value=1.0,
            dtype=tf.float32,  # 明确指定为浮点类型
            trainable=False,
            name="attention_progress"
        )
        self.ema = ExponentialMovingAverage(decay=0.999)

        # 构建模型
        self.build_model()

    def build_model(self):
        # 生成器与判别器的设定
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        # 动态学习率调度（余弦退火）
        self.gen_lr = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1e-5,  # 初始更低
            decay_steps=1000,
            end_learning_rate=1e-4  # 后期恢复原值
        )
        self.disc_lr = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=5e-5,  # 提高初始学习率
            decay_steps=3000,  # 延长衰减步数
            end_learning_rate=1e-6  # 保留微小学习率
        )
        self.gen_optimizer = tf.keras.optimizers.Adam(
            self.gen_lr, beta_1=0.5, beta_2=0.999, clipvalue=10
        )
        # self.gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
        # # 添加EMA封装
        # self.gen_optimizer = LossScaleOptimizer(self.gen_optimizer)
        # self.gen_optimizer = tf.keras.optimizers.MovingAverage(
        #     self.gen_optimizer, average_decay=0.999
        # )
        self.disc_optimizer = tf.keras.optimizers.Adam(self.disc_lr, beta_1=0.5, beta_2=0.9)
        # self.disc_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)  # 更低的学习率

    def build_generator(self):
        inputs = layers.Input(shape=(self.image_size, self.image_size, self.c_dim * self.input_photo_num))

        # 编码器（保持原3层下采样）
        e1 = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
        e1 = layers.LeakyReLU(0.2)(e1)
        e2 = layers.Conv2D(128, (4, 4), strides=2, padding='same')(e1)
        e2 = layers.BatchNormalization()(e2)
        e2 = layers.LeakyReLU(0.2)(e2)
        e3 = layers.Conv2D(256, (4, 4), strides=2, padding='same')(e2)
        e3 = layers.BatchNormalization()(e3)
        e3 = layers.LeakyReLU(0.2)(e3)

        r = e3
        for _ in range(4):
            r = self.residual_block_normal(r)

        # 解码器（减少为2层上采样 + 跳跃连接）
        d1 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(e3)
        d1 = layers.Concatenate()([d1, e2])  # 跳跃连接
        d1 = layers.BatchNormalization()(d1)
        d1 = layers.LeakyReLU(0.2)(d1)

        d2 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same')(d1)
        d2 = layers.Concatenate()([d2, e1])  # 跳跃连接
        d2 = layers.BatchNormalization()(d2)
        d2 = layers.LeakyReLU(0.2)(d2)

        d3 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same')(d2)
        d3 = layers.BatchNormalization()(d3)
        d3 = layers.LeakyReLU(0.2)(d3)

        out = layers.Conv2D(self.c_dim, (3, 3), padding='same', activation='tanh')(d3)
        return models.Model(inputs, out)

    def build_discriminator(self):
        inputs = layers.Input(shape=(self.image_size, self.image_size, self.c_dim))

        # 第一层卷积 + SE注意力
        x = SpectralNormalization(layers.Conv2D(64, (4, 4), strides=2, padding='same'))(inputs)
        # x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        # x = layers.MaxPooling2D(2)(x)
        # x = se_block(x, ratio=8)  # 添加SE模块，ratio为通道压缩比例

        # 第二层卷积 + SE注意力
        x = SpectralNormalization(layers.Conv2D(128, (4, 4), strides=2, padding='same'))(x)
        # x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        # x = layers.MaxPooling2D(2)(x)
        # x = se_block(x, ratio=8)

        # 第三层卷积 + SE注意力
        x = SpectralNormalization(layers.Conv2D(256, (4, 4), strides=2, padding='same'))(x)
        # x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        # x = layers.MaxPooling2D(2)(x)
        # x = se_block(x, ratio=8)

        # 全局池化 + 输出层
        x = layers.GlobalAveragePooling2D()(x)
        outputs = SpectralNormalization(layers.Dense(1))(x)
        # outputs = layers.Dense(1)(x)
        return models.Model(inputs, outputs)

    def residual_block_normal(self, x):
        # 不含特征强化的残差模块：两次卷积然后将原图和卷积结果叠加，注意此处卷积步长默认为1，不改变图像分辨率
        y = layers.Conv2D(self.gf_dim * 4, (3, 3), padding='same')(x)
        y = layers.BatchNormalization()(y)
        # y = InstanceNormalization()(y)
        y = layers.ReLU()(y)

        y = layers.Conv2D(self.gf_dim * 4, (3, 3), padding='same')(y)
        y = layers.BatchNormalization()(y)
        # y = InstanceNormalization()(y)

        return layers.Add()([x, y])

    def compile(self):
        # 生成器和判别器的训练步骤：设定优化器和学习率
        self.generator.compile(loss='binary_crossentropy', optimizer=self.gen_optimizer, metrics=['accuracy'])
        self.discriminator.compile(optimizer=self.disc_optimizer)

    def train_step(self, real_images, hdr_images, epoch):
        # 生成器前向传播
        print("新训练开始")
        with tf.GradientTape() as gen_tape:
            total_epochs = 100
            generated_images = self.generator(real_images, training=True)
            noise_std = tf.maximum(0.01 * (1 - epoch / total_epochs), 0.001)
            generated_images_noisy = generated_images + tf.random.normal(
                tf.shape(generated_images),
                mean=0.0,
                stddev=noise_std  # 噪声强度可调
            )
            fake_output = self.discriminator(generated_images_noisy, training=True)
            adv_loss = -tf.reduce_mean(fake_output)  # 原始对抗损失
            # l1_loss = tf.reduce_mean(tf.abs(hdr_images - generated_images))  # 增强L1约束
            mask = tf.cast(hdr_images < 0.2, tf.float32)
            l1_loss = 0.5 * tf.reduce_mean(tf.abs(hdr_images - generated_images)) + \
                      0.5 * tf.reduce_mean(tf.abs(hdr_images - generated_images) * mask)
            # l1_loss = smooth_l1_loss(hdr_images, fake_output)
            vgg_loss = perceptual_loss(hdr_images, generated_images)
            total_epochs = 100
            # l1_weight = 0.5 + 0.1 * (epoch / total_epochs)  # 从0.8降至0.5
            # adv_weight = 0.4 - 0.1 * (epoch / total_epochs)  # 从0.1升至0.4
            # vgg_weight = 0.4 + 0.1 * (epoch / total_epochs)  # 从0.3升至0.4
            # color_weight = 0.1
            l1_weight = 0.6
            adv_weight = 0.3
            vgg_weight = 0.5
            color_weight = 0.1
            gen_loss = (
                    l1_weight * l1_loss +
                    adv_weight * adv_loss +
                    vgg_weight * vgg_loss +
                    color_weight * color_consistency_loss(hdr_images, generated_images)
            )

        if epoch >= 5:
            train_disc_num = 2
        else:
            train_disc_num = 2
        # 判别器多次更新（独立计算梯度惩罚）
        for _ in range(train_disc_num):
            with tf.GradientTape() as disc_tape:
                real_output = self.discriminator(hdr_images, training=True)
                fake_output = self.discriminator(generated_images_noisy, training=True)

                # 梯度惩罚（强制Lipschitz约束）
                alpha = tf.random.uniform(shape=[tf.shape(real_images)[0], 1, 1, 1])
                interpolates = alpha * hdr_images + (1 - alpha) * generated_images
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolates)
                    pred = self.discriminator(interpolates, training=True)
                gradients = gp_tape.gradient(pred, interpolates)
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)  # 增大惩罚系数
                print("slopes:", slopes)

                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)  # Wasserstein损失，最大化样本差距
                disc_loss += 10.0 * gradient_penalty + 0.01 * tf.reduce_mean(real_output ** 2)  # 梯度惩罚和正则项
                # output_penalty = 0.1 * (tf.reduce_mean(tf.square(real_output)) + tf.reduce_mean(tf.square(fake_output)))
                # disc_loss += output_penalty

            disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            # disc_grads, _ = tf.clip_by_global_norm(disc_grads, clip_norm=0.1)
            self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        # 生成器更新
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        if epoch >= 0:
            self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        print("fake_output_mean:", np.mean(fake_output))
        print("fake_output_median:", np.median(fake_output))
        print("real_output_mean:", np.mean(real_output))
        print("real_output_median:", np.median(real_output))
        print("Gradient penalty mean:", gradient_penalty.numpy())
        print(f"梯度最大值: {tf.reduce_max(slopes).numpy():.4f}")
        print(f"梯度平均数: {tf.math.reduce_mean(slopes).numpy():.4f}")
        ssim_value = tf.reduce_mean(tf.image.ssim(hdr_images, generated_images, max_val=1.0))
        psnr_value = tf.reduce_mean(tf.image.psnr(hdr_images, generated_images, max_val=1.0))
        print(f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}")
        print(f"生成器学习率: {self.gen_optimizer.learning_rate.numpy():.2e}")
        print(f"判别器学习率: {self.disc_optimizer.learning_rate.numpy():.2e}")
        # print("Generator variables:", self.generator.trainable_variables)
        # self.ema.apply(var_list=self.generator.weights)

        return {'gen_loss': gen_loss,
                'disc_loss': disc_loss,
                'l1_loss': l1_loss,
                'ssim': ssim_value,  # 新增
                'psnr': psnr_value,  # 新增
                'grad_penalty': gradient_penalty.numpy(),
                }, generated_images  # 返回值

    def save(self, checkpoint_dir, step):
        # 保存模型（权重）
        checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
        self.generator.save_weights(checkpoint_path + '_generator_{}.weights.h5'.format(step))
        self.discriminator.save_weights(checkpoint_path + '_discriminator_{}.weights.h5'.format(step))

        # 全部保存
        self.generator.save("./model/model_generator_3.h5")
        self.discriminator.save("./model/model_discriminator_3.h5")
        self.generator.save("./model/model_generator_3.keras",
                            save_format='tf',
                            include_optimizer=False
                            )
        self.discriminator.save("./model/model_discriminator_3.keras",
                                save_format='tf',
                                include_optimizer=False
                                )

        # 优化器必须保存，不然从断点加载模型恢复训练必模式崩溃
        # 保存优化器状态
        opt1_vars = self.gen_optimizer.variables  # 获取优化器1的所有变量
        opt2_vars = self.disc_optimizer.variables  # 获取优化器2的所有变量

        # 保存每个优化器的变量
        for var in opt1_vars:
            np.save(f"{checkpoint_dir}/opt/opt1_{step}_{var.name}.npy", var.numpy())

        for var in opt2_vars:
            np.save(f"{checkpoint_dir}/opt/opt2_{step}_{var.name}.npy", var.numpy())

    def load(self, checkpoint_dir, step):
        # 加载模型
        checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
        self.generator.load_weights(checkpoint_path + '_generator_{}.weights.h5'.format(step))
        self.discriminator.load_weights(checkpoint_path + '_discriminator_{}.weights.h5'.format(step))

        print("模型加载完成")

        # 优化器也要导入
        # 加载优化器状态
        opt1_vars = self.gen_optimizer.variables  # 获取优化器1的变量
        opt2_vars = self.disc_optimizer.variables  # 获取优化器2的变量

        # 加载优化器1的变量
        for var in opt1_vars:
            loaded = np.load(f"{checkpoint_dir}/opt/opt1_{step}_{var.name}.npy")
            var.assign(loaded)

        # 加载优化器2的变量
        for var in opt2_vars:
            loaded = np.load(f"{checkpoint_dir}/opt/opt2_{step}_{var.name}.npy")
            var.assign(loaded)

        print("优化器导入完成")

    def load_model_only(self, checkpoint_dir, step):
        # 加载模型
        checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
        self.generator.load_weights(checkpoint_path + '_generator_{}.weights.h5'.format(step))
        self.discriminator.load_weights(checkpoint_path + '_discriminator_{}.weights.h5'.format(step))

        print("模型加载完成")

    def load_model_gen_only(self, checkpoint_dir, step):
        # 加载模型
        checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
        self.generator.load_weights(checkpoint_path + '_generator_{}.weights.h5'.format(step))

        print("模型加载完成")


