from model_simple_etc import *
import data_loader_gan
import matplotlib.pyplot as plt

# 从头开始全局训练，静待佳音
# 注意我说的

# config_set 配置参数
config = {
    'batch_size': 24,           # 批量大小
    'image_size': 256,          # 图像尺寸
    'c_dim': 3,                 # 通道数，通常为RGB图像，所以为3
    'input_photo_num': 7,       # LDR图像数量
    'checkpoint_dir': './check_points',  # 模型检查点路径
    'learning_rate': 0.0002,    # 学习率
    'beta1': 0.5,               # Adam优化器的beta1参数
    'num_epochs': 300,          # 训练的轮数
    'save_interval': 10,        # 每多少轮保存一次模型
    'load_size': 286,           # 输入图像的加载尺寸
    'fine_size': 256,           # 输入图像的目标尺寸（通常为训练图像尺寸）
    'num_shots': 1,             # 曝光次数或其他的训练相关参数
    'dataset': 'VDS',           # 数据集路径
    # 'dataset_hdr': 'VDS/train_hdr_set',    # 数据集路径
}

# 加载模型
hdr_gan = HDR_GAN(config)

# 编译模型
hdr_gan.compile()

hdr_gan.load_model_only(config['checkpoint_dir'], 240)  #当前共计：350轮次，待续
# hdr_gan.load_model_gen_only(config['checkpoint_dir'], 20)  #当前共计：1030轮次，待续

# 数据集
in_p, out_p = data_loader_gan.load_dataset_2(config['dataset'] + '/train_set', config['dataset'] + '/train_hdr_set', image_size=config['image_size'])
dataset = data_loader_gan.create_dataset(in_p, out_p, config['batch_size'])
dataset = dataset.shuffle(config['batch_size'])
tf.compat.v1.enable_eager_execution()

in_p_2, out_p_2 = data_loader_gan.load_dataset_2(config['dataset'] + '/test_set', config['dataset'] + '/test_hdr_set',  image_size=config['image_size'])
dataset_2 = data_loader_gan.create_dataset(in_p_2, out_p_2, config['batch_size'])
dataset_2 = dataset_2.shuffle(config['batch_size'])
tf.compat.v1.enable_eager_execution()

# 近似处理，导入的是.png色调映射后的文件不是.hdr，先留个底

j = 1

# # 训练循环
# for epoch in range(config['num_epochs']):
#     i = 1
#     for batch in dataset:
#         ldr_images, hdr_images = batch
#         # gen_loss, disc_loss = hdr_gan.train_step(ldr_images, hdr_images)
#         loss = hdr_gan.train_step_Beta(ldr_images, hdr_images)
#         # print(gen_loss, disc_loss, i, j)
#         print(loss['gen_loss'], loss['disc_loss'], loss['l1_loss'], i, j)
#         i += 1
#
#     # 每一定步长保存模型
#     if (epoch + 1) % config['save_interval'] == 0:
#         hdr_gan.save(config['checkpoint_dir'], epoch + 1)
#     j += 1


# 计算每个 epoch 的步数
steps_per_epoch = 48 // config['batch_size']

# hdr_gan.reset_optimizers()
# hdr_gan.compile()
# hdr_gan.load_model_only(config['checkpoint_dir'], 80)
# hdr_gan.discriminator = hdr_gan.build_discriminator()

# 训练日记：
# 0-100：渐进损失函数训练（0.7->0.3, 0.3->0.7, 0.01），梯度惩罚1.0
# 100-150：锁定损失函数训练：0.5:0.5:0.01，梯度惩罚0.1
# 150-350：锁定损失函数训练：0.4:0.4:0.2，梯度惩罚10.0
# 350-550：新增生成器损失函数：多尺度感知损失、色调映射损失：0.1:0.3:0.5:0.1
# 550-1030：梯度惩罚权重500，损失不变，拉低判别器学习率
# 1030-1480：重启判别器，判别器架构大改移除谱归一化释放梯度（效果不好已取消）
# 1030-1280：数据集小更新，新增输入输出直方图均衡，试图破坏黑影主导地位，锁定损失函数配比：0.1:0.4:0.4:0.1
# 1280-1480：梯度惩罚正常化10.0，判别器损失函数取消正则化
# 1480-1780：数据集大更新，新增乱序输入增强数据，破坏伪收敛过拟合平衡重新收敛，修正损失函数配比：0.6:0.2:0.1:0.1 -> 0.2:0.6:0.1:0.1
# 1780-：正式破局，验证集上数据增强效果显著，损失函数配比重构：0.2:0.4:0.3:0.1

print("GPU列表:", tf.config.list_physical_devices('GPU'))
print("TensorFlow版本:", tf.__version__)
g_loss = []
d_loss = []

for epoch in range(0, config['num_epochs']):
    i = 1
    for _ in range(steps_per_epoch):  # 每个 epoch 只迭代 steps_per_epoch 次
        ldr_images, hdr_images = next(iter(dataset))  # 手动获取下一个批次
        loss, image_output = hdr_gan.train_step(ldr_images, hdr_images, epoch)
        print("gen_loss:", loss['gen_loss'], "disc_loss:", loss['disc_loss'], "l1_loss:", loss['l1_loss'], i, epoch + 1)
        i += 1
        g_loss.append(loss['gen_loss'])
        d_loss.append(loss['disc_loss'])

    ldr_images, hdr_images = next(iter(dataset_2))
    image_output_2 = hdr_gan.generator(ldr_images)
    data_loader_gan.display_hdr(image_output_2[0])

    plt.subplot(121)
    plt.plot(g_loss)
    plt.subplot(122)
    plt.plot(d_loss)
    plt.show()

    # 每一定步长保存模型
    if (epoch + 1) % config['save_interval'] == 0:
        hdr_gan.save(config['checkpoint_dir'], epoch + 1)
        in_p, out_p = data_loader_gan.load_dataset_2(config['dataset'] + '/train_set',
                                                     config['dataset'] + '/train_hdr_set')
        dataset = data_loader_gan.create_dataset(in_p, out_p, config['batch_size'])
        dataset = dataset.shuffle(config['batch_size'])  # 重置数据集
        tf.compat.v1.enable_eager_execution()
