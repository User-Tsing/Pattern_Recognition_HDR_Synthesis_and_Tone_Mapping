import numpy as np
import tensorflow as tf
import os
import glob
import imageio
import matplotlib.pyplot as plt
import cv2
import random


def clahe_preprocess(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    自适应直方图均衡化（CLAHE）
    参数：
      image: 输入图像（TensorFlow Tensor，范围[0,1]或[0,255]）
      clip_limit: 对比度限制阈值
      tile_grid_size: 局部区域大小
    """
    # 转换图像为numpy数组并调整范围到[0,255]
    image_np = image.numpy()
    image_np = (image_np * 255).astype(np.uint8) if image_np.max() <= 1.0 else image_np.astype(np.uint8)

    # 转换为LAB颜色空间，仅对亮度通道(L)做CLAHE
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # 归一化到[-1,1]（与生成器输入范围匹配）
    image_clahe = image_clahe.astype(np.float32) / 127.5 - 1.0
    return image_clahe

def clahe_enhance(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对256x256x3的RGB图像进行CLAHE增强
    参数：
      image: 输入图像（numpy数组，范围[0, 255]或[0, 1]）
      clip_limit: 对比度限制阈值（默认2.0）
      tile_grid_size: 局部区域大小（默认8x8）
    返回：
      enhanced_image: 增强后的RGB图像（范围与输入一致）
    """
    # 确保输入为[0, 255]的uint8类型
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # 转换为LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 对L通道应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    # 合并通道并转回RGB
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # 保持与输入一致的范围
    if image.max() <= 1.0:
        enhanced_image = enhanced_image.astype(np.float32) / 255.0
    return enhanced_image

def load_image(image_path, size=256):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(size, size))
    img = tf.keras.preprocessing.image.img_to_array(img)
    return img

# def load_hdr_image(image_path, size=256):
#     # img = tf.keras.preprocessing.image.load_img(image_path, target_size=(size, size))
#     # img = tf.keras.preprocessing.image.img_to_array(img)
#     img = imageio.imread(image_path)
#     img = img.astype(np.float32)
#     img_resized = tf.image.resize(img, (size, size), method='bilinear')
#     img_resized = tf.keras.preprocessing.image.img_to_array(img_resized)
#     return img_resized

def load_hdr_image(image_path, size=256, low_percentile=0.1, high_percentile=99.9, gamma=0.7):
    """
    科学加载HDR图像，适配模型训练
    参数：
      low_percentile: 下分位数截断（建议0.1-1）
      high_percentile: 上分位数截断（建议99-99.9）
      gamma: 暗部增强系数（0.5-1.2）
    """
    # 读取原始HDR数据（保留浮点精度）
    img = imageio.imread(image_path).astype(np.float32)
    img = tf.image.resize(img, (size, size), method='bilinear').numpy()

    # 分位数截断（排除极端噪声/过曝）
    p_low = np.percentile(img, low_percentile)
    p_high = np.percentile(img, high_percentile)
    img = np.clip(img, p_low, p_high)

    # 归一化到[0,1]（基于截断后范围）
    img = (img - p_low) / (p_high - p_low + 1e-7)

    # 自适应Gamma校正（增强暗部细节）
    img = np.power(img, gamma)

    # 映射到模型输入范围[-1,1]
    img = img * 2.0 - 1.0
    return img

def load_hdr_image_old(image_path, size=256):
    # 1. 读取HDR图像，并确保它是浮动格式（32位）
    img = imageio.imread(image_path).astype(np.float32)

    # 2. 对数色调映射：避免极大或极小的值导致问题
    img = np.log1p(img)  # log(1+x)避免数值问题

    # 3. 归一化到[-1, 1]范围
    mean, std = np.mean(img), np.std(img)
    img = (img - mean) / (std + 1e-7)
    img = np.clip(img, -3.0, 3.0) / 3.0  # 保留99%的数据分布

    # 4. 调整图像尺寸（双线性插值）
    img_resized = tf.image.resize(img, (size, size), method='bilinear')

    # 5. 反归一化，准备显示
    image_display = (img_resized.numpy() + 1) / 2  # 将数据从[-1, 1]恢复到[0, 1]范围

    # 显示图像
    # plt.imshow(image_display)
    # plt.axis('off')  # 可选：隐藏坐标轴
    # plt.show()

    return img_resized
def load_dataset(image_path, img_path_hdr, batch_size=6, image_size=256):
    input_data = []
    output_data = []
    sd = 0
    # dataset = tf.data.Dataset.list_files(image_path)
    for subdir in os.listdir(image_path):
        subdir_path = os.path.join(image_path, subdir)
        print(subdir_path)
        sd += 1
        if os.path.isdir(subdir_path):
            photo = []
            subdir_path_hdr = os.path.join(img_path_hdr, subdir)
            for image_file in glob.glob(os.path.join(subdir_path, '*.png')):
                image_input = load_image(image_file)
                image_input = image_input / 127.5 - 1  # 归一化
                print(image_input)
                photo.append(image_input)
            # hdr_image = load_hdr_image(os.path.join(subdir_path_hdr, 'target_hdr.hdr'))
            hdr_image = load_image(os.path.join(subdir_path_hdr, 'tonemap.png'))
            hdr_image = hdr_image / 127.5 - 1
            # print(hdr_image)
            input_data.append(tf.concat(photo, axis=-1))
            output_data.append(hdr_image)
        print(sd)
    print(len(input_data))
    return np.array(input_data), np.array(output_data)

def load_dataset_2(image_path, img_path_hdr, batch_size=6, image_size=256):
    input_data = []
    output_data = []
    sd = 0
    # dataset = tf.data.Dataset.list_files(image_path)
    for subdir in os.listdir(image_path):
        subdir_path = os.path.join(image_path, subdir)
        print(subdir_path)
        sd += 1
        if os.path.isdir(subdir_path):
            photo = []
            subdir_path_hdr = os.path.join(img_path_hdr, subdir)
            img_file = list(glob.glob(os.path.join(subdir_path, '*.png')))
            random.shuffle(img_file)
            for image_file in img_file:
                image_input = load_image(image_file, image_size)
                enhanced_np = clahe_enhance(image_input, clip_limit=2.0)
                image_input = enhanced_np / 127.5 - 1  # 归一化
                # print(image_input)
                photo.append(image_input)
            # hdr_image = load_hdr_image(os.path.join(subdir_path_hdr, 'target_hdr.hdr'))
            # hdr_image = load_hdr_image(os.path.join(subdir_path_hdr, 'target_hdr.hdr'), image_size)
            hdr_image = load_hdr_image(
                os.path.join(subdir_path_hdr, 'target_hdr.hdr'),
                size=image_size,
                low_percentile=0.1,
                high_percentile=99.9,
                gamma=0.7
            )
            print(np.min(hdr_image), np.max(hdr_image))
            # print(hdr_image)
            input_data.append(tf.concat(photo, axis=-1))
            output_data.append(hdr_image)
            if len(input_data) % 10 == 0:  # 每10个样本可视化一次
                display_hdr(hdr_image)
        print(sd)
    print(len(input_data))
    return np.array(input_data), np.array(output_data)

def create_dataset(input_data, output_data, batch_size=6):
    dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
    dataset = dataset.shuffle(buffer_size=len(input_data))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset



def load_dataset_val(image_path, batch_size=6, image_size=256):
    input_data = []
    output_data = []
    sd = 0
    # dataset = tf.data.Dataset.list_files(image_path)
    for subdir in os.listdir(image_path):
        subdir_path = os.path.join(image_path, subdir)
        print(subdir_path)
        sd += 1
        if os.path.isdir(subdir_path):
            photo = []
            for image_file in glob.glob(os.path.join(subdir_path, '*aligned.tif')):
                image_input = load_image(image_file)
                image_input = image_input / 127.5 - 1  # 归一化
                photo.append(image_input)
                photo.append(image_input)
            hdr_image = load_hdr_image(os.path.join(subdir_path, 'ref_hdr_aligned.hdr'))
            hdr_image = hdr_image / 127.5 - 1
            input_data.append(tf.concat(photo, axis=-1))
            output_data.append(hdr_image)
        print(sd)
    print(len(input_data))
    return np.array(input_data), np.array(output_data)


def display_hdr(hdr_image):
    """将HDR训练数据转换为可显示的8位图像"""
    # 1. 反归一化到[0,1]
    img_linear = (hdr_image + 1.0) / 2.0

    # 2. 应用ACES色调映射（保留高光细节）
    aces_curve = lambda x: (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    img_tonemapped = aces_curve(img_linear)

    # 3. Gamma校正（适配sRGB显示）
    img_gamma = np.power(img_tonemapped, 1 / 2.2)

    # 4. 转换为8位并显示
    img_8bit = (np.clip(img_gamma, 0, 1) * 255).astype(np.uint8)
    plt.imshow(img_8bit)
    plt.axis('off')
    plt.show()