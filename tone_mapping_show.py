import sys
import os
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt
from matplotlib import pyplot as plt
import imageio
import cv2
import data_loader_gan
from model_simple_etc import HDR_GAN
from data_loader_gan import clahe_enhance, load_image

import tone_mapping_delicated_show

config = {
    'batch_size': 6,
    'image_size': 512,
    'c_dim': 3,
    'input_photo_num': 7,
    'checkpoint_dir': './check_points'
}

def resource_path(relative_path):
    # 为配合可能打包组装.exe文件时寻址可能产生的问题，定义路径寻址方法
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS  # 临时解压目录
    else:
        base_path = os.path.abspath(".")  # 开发环境目录
    return os.path.join(base_path, relative_path)


class HDRGANApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_images = []
        self.current_idx = 0
        self.input_folder = ""
        self.detail_text_show = "模式识别 + 数字图像处理"
        self.initUI()
        self.load_model()
        self.tmo_results = []
        self.performance = []
        self.current_tmo_idx = 0
        self.aligned_images = []

    def initUI(self):
        self.setWindowTitle('HDR Image Generator Pro')
        self.setGeometry(100, 100, 1400, 850)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.setFixedSize(1400, 850)
        self.setWindowIcon(QIcon(resource_path("icon/photo.ico")))

        self.font_1 = QFont('Arial', 25)
        self.font_1.setBold(True)
        self.font_2 = QFont('Arial', 14)
        self.font_2.setBold(True)
        self.font_3 = QFont('Arial', 8)
        self.font_3.setBold(True)

        self.title_text = QLabel(self)
        self.title_text.setText("HDR合成与色调映射效果展示")
        self.title_text.setAlignment(Qt.AlignCenter)
        self.title_text.setFont(self.font_1)
        self.title_text.setStyleSheet("color: rgb(55, 0, 155);")
        self.title_text.setGeometry(0, 0, 1400, 50)

        self.label_detail = QLabel(self)
        self.label_detail.setText(self.detail_text_show)
        self.label_detail.setAlignment(Qt.AlignCenter)
        self.label_detail.setFont(self.font_2)
        self.label_detail.setStyleSheet("color: rgb(55, 0, 155);")
        self.label_detail.setGeometry(0, 50, 1400, 50)

        # 图像显示区域
        self.input_label = QLabel(self)
        self.output_label = QLabel(self)
        self.input_label.setGeometry(150, 100, 512, 512)
        self.input_label.setStyleSheet("background-color: rgb(55, 0, 155);")
        self.output_label.setGeometry(1400-662, 100, 512, 512)
        self.output_label.setStyleSheet("background-color: rgb(55, 0, 155);")

        # 按钮
        self.align_btn = QPushButton('对齐图像', self)
        self.align_btn.setGeometry(380, 650, 300, 50)
        self.align_btn.setFont(self.font_2)
        self.load_hdr_btn = QPushButton('导入HDR并色调映射', self)
        self.load_hdr_btn.setGeometry(40, 720, 300, 50)
        self.load_hdr_btn.setFont(self.font_2)
        self.tmo_prev_btn = QPushButton('上一个TMO', self)
        self.tmo_prev_btn.setGeometry(380, 720, 300, 50)
        self.tmo_prev_btn.setFont(self.font_2)
        self.tmo_next_btn = QPushButton('下一个TMO', self)
        self.tmo_next_btn.setGeometry(720, 720, 300, 50)
        self.tmo_next_btn.setFont(self.font_2)
        self.spectrum_btn = QPushButton('显示频谱', self)
        self.spectrum_btn.setGeometry(1060, 720, 300, 50)
        self.spectrum_btn.setFont(self.font_2)
        self.next_btn = QPushButton('下一张输入图', self)
        self.next_btn.setGeometry(1060, 650, 300, 50)
        self.next_btn.setFont(self.font_2)
        self.import_btn = QPushButton('导入图像', self)
        self.import_btn.setGeometry(40, 650, 300, 50)
        self.import_btn.setFont(self.font_2)
        self.predict_btn = QPushButton('生成HDR', self)
        self.predict_btn.setGeometry(720, 650, 300, 50)
        self.predict_btn.setFont(self.font_2)

        self.label_maker = QLabel(self)
        self.label_maker.setText("Directed by STAssn (@STAssn_2020)")
        self.label_maker.setFont(self.font_3)
        self.label_maker.setStyleSheet("color: rgb(55, 0, 155);")
        self.label_maker.setGeometry(0, 820, 1400, 30)
        self.label_maker.setAlignment(Qt.AlignCenter)

        # 信号连接
        self.next_btn.clicked.connect(self.show_next_image)
        self.import_btn.clicked.connect(self.load_images)
        self.predict_btn.clicked.connect(self.generate_hdr)
        self.align_btn.clicked.connect(self.align_images)
        self.load_hdr_btn.clicked.connect(self.load_hdr_and_tone_map)
        self.tmo_prev_btn.clicked.connect(self.show_prev_tmo)
        self.tmo_next_btn.clicked.connect(self.show_next_tmo)
        self.spectrum_btn.clicked.connect(self.show_spectrum)

        self.open_another_interface = QPushButton(self)
        self.open_another_interface.setGeometry(40, 790, 300, 50)
        self.open_another_interface.setText("Reinhard可变参数色调映射")
        self.open_another_interface.setFont(self.font_2)
        self.open_another_interface.clicked.connect(self.open_another_interface_clicked)

    # 图像对齐功能
    def align_images(self):
        if not self.input_images:
            return

        ref_image = (self.input_images[0] + 1) * 127.5
        ref_image = ref_image.astype(np.uint8)
        self.aligned_images = [ref_image]

        # 使用SIFT进行特征匹配
        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(ref_image, None)

        for img in self.input_images[1:]:
            img_uint8 = ((img + 1) * 127.5).astype(np.uint8)
            kp_img, des_img = sift.detectAndCompute(img_uint8, None)

            # FLANN匹配器
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des_ref, des_img, k=2)

            # 筛选优质匹配
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # 计算单应性矩阵
            if len(good) > 10:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                aligned = cv2.warpPerspective(img_uint8, M, (512, 512))
                self.aligned_images.append(aligned)

        # 更新输入图像
        self.input_images = [(img.astype(np.float32) / 127.5 - 1) for img in self.aligned_images]
        self.current_idx = 0
        self.update_input_display()
        self.detail_text_show = "模式识别 + 数字图像处理： 图像对齐"
        self.label_detail.setText(self.detail_text_show)

    # 色调映射功能
    def tone_mapping(self, hdr_path):
        # 色调映射多种方法叠加
        try:
            hdr = imageio.imread(hdr_path).astype(np.float32)
            hdr_rgb = cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR)
            hdr = cv2.normalize(hdr_rgb, None, 0.0, 1.0, cv2.NORM_MINMAX)
            hdr_2 = self.robust_normalize(hdr_rgb)

            # 模型训练时的归一化与色调映射方法
            model_trainer = self.tone_mapping_model_trainer(hdr_rgb)

            # 原始三种方法
            reinhard = cv2.createTonemapReinhard(2.2, 0.5, 0.5, 0.5)
            tmo_reinhard = reinhard.process(hdr.copy())
            print(tmo_reinhard.shape)

            drago = cv2.createTonemapDrago(1.0, 0.7)
            tmo_drago = drago.process(hdr.copy())

            # 新增Mantiuk算法
            mantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 0.85)
            tmo_mantiuk = mantiuk.process(hdr.copy())

            # 新增Exposure Fusion
            exposures = [-6, 0, +6]  # 曝光参数
            fusion = self.exposure_fusion(hdr, exposures)  # 这个算了效果真不行

            aces = self.aces_filmic(hdr_2)

            # 统一后处理
            results = []
            for tmo in [tmo_reinhard, tmo_drago, tmo_mantiuk, aces, model_trainer]:
                img = (tmo * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results.append(img)

            # 计算性能指标
            metrics = [self.calculate_image_metrics(img) for img in results]

            return results, metrics

        except Exception as e:
            print(f"色调映射错误: {str(e)}")
            return [], []

    def robust_normalize(self, hdr):
        # 归一化方法之一
        # 计算分位数
        p_low = np.percentile(hdr, 0.1)
        p_high = np.percentile(hdr, 99.9)

        # 线性归一化
        hdr_clipped = np.clip(hdr, p_low, p_high)
        return cv2.normalize(hdr_clipped, None, 0.0, 1.0, cv2.NORM_MINMAX)

    def exposure_fusion(self, hdr, exposures):
        # 多曝光融合，效果不佳未启用
        # 生成多曝光序列
        exposures = [cv2.convertScaleAbs(hdr, alpha=10 ** e) for e in exposures]

        # 创建融合器
        merge_mertens = cv2.createMergeMertens()
        fusion = merge_mertens.process(exposures)
        return fusion

    def aces_filmic(self, hdr):
        # ACES色调映射方法
        # 调整输入范围
        # x = hdr * 0.8  # 曝光调整
        x = hdr

        # ACES曲线
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return (x * (a * x + b)) / (x * (c * x + d) + e)

    def tone_mapping_model_trainer(self, img, low_percentile=0.1, high_percentile=99.9, gamma=0.7):
        # 与模型训练时等效的色调映射方法
        # 分位数截断（排除极端噪声/过曝）
        p_low = np.percentile(img, low_percentile)
        p_high = np.percentile(img, high_percentile)
        img = np.clip(img, p_low, p_high)

        # 归一化到[0,1]（基于截断后范围）
        img = (img - p_low) / (p_high - p_low + 1e-7)

        # 自适应Gamma校正（增强暗部细节）
        img = np.power(img, gamma)

        # 应用ACES色调映射（保留高光细节）
        aces_curve = lambda x: (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
        img_tonemapped = aces_curve(img)

        # Gamma校正（适配sRGB显示）
        img_gamma = np.power(img_tonemapped, 1 / 2.2)

        return img_gamma

    def calculate_image_metrics(self, image):
        # 色调映射效果量化指标测评
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 对比度（标准差）
        contrast = np.std(gray)

        # 信息熵
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel() / gray.size
        entropy = -np.sum(hist * np.log2(hist + 1e-7))

        # 平均梯度
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        avg_gradient = np.mean(np.sqrt(gx ** 2 + gy ** 2))

        # 动态范围利用率
        min_val, max_val = gray.min(), gray.max()
        dynamic_range = (max_val - min_val) / 255.0

        return {
            '对比度': round(contrast, 2),
            '信息熵': round(entropy, 2),
            '平均梯度': round(avg_gradient, 2),
            '动态范围': round(dynamic_range, 2)
        }

    # 在界面类中新增导出方法，暂未启用
    def export_metrics(self):
        if not hasattr(self, 'tmo_metrics') or not self.tmo_metrics:
            return

        path, _ = QFileDialog.getSaveFileName(self, '保存指标', '', 'CSV文件 (*.csv)')
        if path:
            import csv
            methods = ['Reinhard', 'Drago', 'Mantiuk', 'ExposureFusion']

            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['方法', '对比度', '信息熵', '平均梯度', '动态范围'])
                for method, metric in zip(methods, self.tmo_metrics):
                    writer.writerow([
                        method,
                        metric['对比度'],
                        metric['信息熵'],
                        metric['平均梯度'],
                        metric['动态范围']
                    ])

    def load_hdr_and_tone_map(self):
      # 导入.hdr文件
        path, _ = QFileDialog.getOpenFileName(self, '选择HDR文件', '', 'HDR文件 (*.hdr)')
        if path:
            self.tmo_results, self.performance = self.tone_mapping(path)
            self.current_tmo_idx = 0
            self.update_tmo_display()
            self.detail_show()

    def detail_show(self):
        if self.tmo_results:
            if self.current_tmo_idx == 0:
                self.detail_text_show = "模式识别 + 数字图像处理：色调映射Reinhard，"
            elif self.current_tmo_idx == 1:
                self.detail_text_show = "模式识别 + 数字图像处理：色调映射Drago，"
            elif self.current_tmo_idx == 2:
                self.detail_text_show = "模式识别 + 数字图像处理：色调映射Mantiuk，"
            elif self.current_tmo_idx == 3:
                self.detail_text_show = "模式识别 + 数字图像处理：色调映射ACES，"
            elif self.current_tmo_idx == 4:
                self.detail_text_show = "模式识别 + 数字图像处理：基于ACES和Gamma校正的综合映射，"
            else:
                self.detail_text_show = "模式识别 + 数字图像处理：色调映射，"
            self.detail_text_show += "对比度:"
            self.detail_text_show += str(self.performance[self.current_tmo_idx]['对比度'])
            self.detail_text_show += "，信息熵:"
            self.detail_text_show += str(self.performance[self.current_tmo_idx]['信息熵'])
            self.detail_text_show += "，平均梯度:"
            self.detail_text_show += str(self.performance[self.current_tmo_idx]['平均梯度'])
            self.detail_text_show += "，动态范围:"
            self.detail_text_show += str(self.performance[self.current_tmo_idx]['动态范围'])
            self.detail_text_show += "。"
            self.label_detail.setText(self.detail_text_show)

    def update_tmo_display(self):
        if self.tmo_results:
            # 获取当前色调映射结果
            img = self.tmo_results[self.current_tmo_idx]

            # 确保内存连续性和正确尺寸
            img = cv2.resize(img, (512, 512))
            img = np.ascontiguousarray(img)

            # 转换QImage并显示
            qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.input_label.setPixmap(QPixmap.fromImage(qimg))

            # 计算频谱
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-7)

            # 归一化到0-255范围
            magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_uint8 = magnitude_normalized.astype(np.uint8)
            magnitude_uint8 = np.ascontiguousarray(magnitude_uint8)

            # 创建QImage并显示
            qimg_fft = QImage(magnitude_uint8.data, 512, 512, QImage.Format_Grayscale8)
            self.output_label.setPixmap(QPixmap.fromImage(qimg_fft))

    def show_prev_tmo(self):
        if self.tmo_results:
            self.current_tmo_idx = (self.current_tmo_idx - 1) % len(self.tmo_results)
            self.update_tmo_display()
            self.detail_show()

    def show_next_tmo(self):
        if self.tmo_results:
            self.current_tmo_idx = (self.current_tmo_idx + 1) % len(self.tmo_results)
            self.update_tmo_display()
            self.detail_show()

    # 频谱显示功能
    def show_spectrum(self):
        if self.tmo_results:
            img = self.tmo_results[self.current_tmo_idx]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-7)

            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(img)
            plt.subplot(122), plt.imshow(magnitude, cmap='gray')
            plt.show()

    def load_model(self):
        self.model = HDR_GAN(config)
        # self.model.generator.build(input_shape=(None, 256, 256, 21))
        # self.model.generator.load_weights('./model/model_generator_3.weights.h5')
        self.model.load_model_gen_only(resource_path(config['checkpoint_dir']), 240)

    def load_images(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Image Folder')
        if folder:
            self.input_folder = folder
            image_files = sorted(
                [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])[:7]

            # 加载并预处理图像
            self.input_images = []
            repeat_image = None
            i = 0
            for f in image_files:
                img = load_image(os.path.join(folder, f), config['image_size'])
                enhanced = clahe_enhance(img, clip_limit=2.0)
                normalized = enhanced / 127.5 - 1
                if i == 0:
                    repeat_image = normalized
                    i += 1
                self.input_images.append(normalized)

            # 零填充不足7张的情况
            while len(self.input_images) < 7:
                # self.input_images.append(np.zeros_like(self.input_images[0]))
                self.input_images.append(repeat_image)

            self.current_idx = 0
            self.update_input_display()

    def preprocess_input(self):
        # 合并输入图像通道
        combined = np.concatenate(self.input_images, axis=-1)
        return np.expand_dims(combined, axis=0)

    def generate_hdr(self):
        if not self.input_images:
            return

        # 生成预测
        model_input = self.preprocess_input()
        prediction = self.model.generator.predict(model_input)[0]

        # 后处理
        hdr_output = (prediction + 1.00) * 0.50  # 转换到[0,1]
        self.show_hdr_image(hdr_output)
        # data_loader_gan.display_hdr(prediction)

        # 保存HDR文件
        # save_path, _ = QFileDialog.getSaveFileName(self, 'Save HDR File', '', 'HDR Files (*.hdr)')
        # if save_path:
        #     linear_hdr = (hdr_output * 65535).astype(np.uint16)
        #     imageio.imwrite(save_path, linear_hdr, format='hdr')

        hdr_output = np.power(hdr_output, 1.0 / 0.7)
        save_path, _ = QFileDialog.getSaveFileName(self, '保存HDR文件', '', 'HDR文件 (*.hdr)')
        if save_path:
            imageio.imwrite(save_path, hdr_output.astype(np.float32), format='hdr')

    def show_hdr_image(self, hdr_array):
        # 色调映射显示
        tonemapped = np.power(hdr_array, 1 / 2.2)
        qimg = QImage((tonemapped * 255).astype(np.uint8), config['image_size'], config['image_size'],
                      QImage.Format_RGB888)
        self.output_label.setPixmap(QPixmap.fromImage(qimg))

    def update_input_display(self):
        if self.input_images:
            img = (self.input_images[self.current_idx] + 1) * 127.5
            qimg = QImage(img.astype(np.uint8), config['image_size'], config['image_size'], QImage.Format_RGB888)
            self.input_label.setPixmap(QPixmap.fromImage(qimg))

    def show_next_image(self):
        if self.input_images:
            self.current_idx = (self.current_idx + 1) % len(self.input_images)
            self.update_input_display()

    def open_another_interface_clicked(self):
        self.new_one = tone_mapping_delicated_show.ToneMappingApp()
        self.new_one.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HDRGANApp()
    ex.show()
    sys.exit(app.exec_())
