import sys
import numpy as np
import cv2
import imageio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QSlider, QFileDialog, QVBoxLayout, QHBoxLayout)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt


class ToneMappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.hdr_image = None
        self.detail_text_show = "数字图像处理"
        self.initUI()
        self.initParameters()

    def initUI(self):
        self.setWindowTitle('HDR实时色调映射')
        self.setGeometry(450, 50, 1200, 850)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.setFixedSize(1200, 850)

        self.font_1 = QFont('Arial', 25)
        self.font_1.setBold(True)
        self.font_2 = QFont('Arial', 14)
        self.font_2.setBold(True)
        self.font_3 = QFont('Arial', 8)
        self.font_3.setBold(True)

        self.label_title = QLabel(self)
        self.label_title.setText("基于Reinhard的HDR可变参数色调映射效果展示")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setFont(self.font_1)
        self.label_title.setStyleSheet("color: rgb(55, 0, 155);")
        self.label_title.setGeometry(0, 0, 1200, 50)

        self.label_detail = QLabel(self)
        self.label_detail.setText(self.detail_text_show)
        self.label_detail.setAlignment(Qt.AlignCenter)
        self.label_detail.setFont(self.font_2)
        self.label_detail.setStyleSheet("color: rgb(55, 0, 155);")
        self.label_detail.setGeometry(0, 50, 1200, 50)

        # 图像显示区域
        self.image_label = QLabel(self)
        # self.image_label.setFixedSize(512, 512)
        self.image_label.setGeometry(20, 100, 512, 512)
        self.image_label.setStyleSheet("background-color: rgb(55, 0, 155);")
        self.spectrum_label = QLabel(self)
        # self.spectrum_label.setFixedSize(512, 512)
        self.spectrum_label.setGeometry(1200-532, 100, 512, 512)
        self.spectrum_label.setStyleSheet("background-color: rgb(55, 0, 155);")

        # 控制面板
        self.createSliders()

        # 导入按钮
        self.import_btn = QPushButton('导入HDR图像', self)
        self.import_btn.clicked.connect(self.loadHDR)
        self.import_btn.setFont(self.font_2)
        self.import_btn.setGeometry(40, 790, 300, 50)

        self.label_maker = QLabel(self)
        self.label_maker.setText("Directed by STAssn (@STAssn_2020)")
        self.label_maker.setFont(self.font_3)
        self.label_maker.setStyleSheet("color: rgb(55, 0, 155);")
        self.label_maker.setGeometry(0, 820, 1200, 30)
        self.label_maker.setAlignment(Qt.AlignCenter)

    def initParameters(self):
        # 初始化映射参数
        self.params = {
            'gamma': 2.2,
            'intensity': 0.5,
            'light_adapt': 0.8,
            'color_adapt': 0.5,
            'saturation': 0.7
        }

    def createSliders(self):
        # 创建参数滑块
        layout_1 = (100, 650, 300, 20, 100, 680, 300, 20)
        self.gamma_slider = self.createSlider("Gamma校正", 1.0, 3.0, 2.2, layout_1)
        layout_2 = (450, 650, 300, 20, 450, 680, 300, 20)
        self.intensity_slider = self.createSlider("映射强度", 0.0, 1.0, 0.5, layout_2)
        layout_3 = (800, 650, 300, 20, 800, 680, 300, 20)
        self.light_slider = self.createSlider("光照适应", 0.0, 1.0, 0.8, layout_3)
        layout_4 = (250, 720, 300, 20, 250, 750, 300, 20)
        self.color_slider = self.createSlider("颜色适应", 0.0, 1.0, 0.5, layout_4)
        layout_5 = (650, 720, 300, 20, 650, 750, 300, 20)
        self.saturation_slider = self.createSlider("饱和度", 0.0, 1.0, 0.7, layout_5)

    def createSlider(self, title, min_val, max_val, init_val, layout):
        slider = QSlider(Qt.Horizontal, self)
        slider.setMinimum(int(min_val * 100))
        slider.setMaximum(int(max_val * 100))
        slider.setValue(int(init_val * 100))
        slider.valueChanged.connect(self.updateToneMapping)

        label = QLabel(f"{title}: {init_val:.2f}", self)

        slider.setGeometry(layout[0], layout[1], layout[2], layout[3])
        label.setGeometry(layout[4], layout[5], layout[6], layout[7])

        slider.label = label
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }

            QSlider::handle:horizontal {
                background: QLinearGradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0; 
                border-radius: 3px;
            }
        """)
        return slider

    def loadHDR(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择HDR文件', '', 'HDR文件 (*.hdr)')
        if path:
            self.hdr_image = imageio.imread(path).astype(np.float32)
            self.updateToneMapping()

    def updateToneMapping(self):
        if self.hdr_image is None:
            return

        # 更新参数值
        self.params['gamma'] = self.gamma_slider.value() / 100
        self.params['intensity'] = self.intensity_slider.value() / 100
        self.params['light_adapt'] = self.light_slider.value() / 100
        self.params['color_adapt'] = self.color_slider.value() / 100
        self.params['saturation'] = self.saturation_slider.value() / 100

        # 更新滑块标签
        self.gamma_slider.label.setText(f"Gamma校正: {self.params['gamma']:.2f}")
        self.intensity_slider.label.setText(f"映射强度: {self.params['intensity']:.2f}")
        self.light_slider.label.setText(f"光照适应: {self.params['light_adapt']:.2f}")
        self.color_slider.label.setText(f"颜色适应: {self.params['color_adapt']:.2f}")
        self.saturation_slider.label.setText(f"饱和度: {self.params['saturation']:.2f}")

        # 执行色调映射
        mapped = self.applyToneMapping()
        self.displayImage(mapped)
        self.displaySpectrum(mapped)
        result = self.calculate_image_metrics(mapped)
        self.detail_show(result)

    def applyToneMapping(self):
        # 使用Reinhard算法进行色调映射
        hdr_norm = cv2.normalize(self.hdr_image, None, 0.0, 1.0, cv2.NORM_MINMAX)
        tonemap = cv2.createTonemapReinhard(
            gamma=self.params['gamma'],
            intensity=self.params['intensity'],
            light_adapt=self.params['light_adapt'],
            color_adapt=self.params['color_adapt']
        )
        ldr = tonemap.process(hdr_norm)

        # 调整饱和度
        hsv = cv2.cvtColor(ldr, cv2.COLOR_BGR2HSV)
        hsv[..., 1] *= self.params['saturation']
        ldr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return (ldr * 255).astype(np.uint8)

    def displayImage(self, image):
        # 显示RGB图像
        image = cv2.resize(image, (512, 512))
        qimg = QImage(image.data, 512, 512, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def displaySpectrum(self, image):
        # 计算并显示频谱
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(gray)
        fshift = np.fft.fftshift(fft)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-7)

        # 归一化处理
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = magnitude.astype(np.uint8)

        # 创建QImage
        qimg = QImage(magnitude.data, 512, 512, QImage.Format_Grayscale8)
        self.spectrum_label.setPixmap(QPixmap.fromImage(qimg))

    def calculate_image_metrics(self, image):
        # 色调映射参数
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

    def detail_show(self, result):
        self.detail_text_show = "数字图像处理：Reinhard色调映射："
        self.detail_text_show += "对比度:"
        self.detail_text_show += str(result['对比度'])
        self.detail_text_show += "，信息熵:"
        self.detail_text_show += str(result['信息熵'])
        self.detail_text_show += "，平均梯度:"
        self.detail_text_show += str(result['平均梯度'])
        self.detail_text_show += "，动态范围:"
        self.detail_text_show += str(result['动态范围'])
        self.detail_text_show += "。"
        self.label_detail.setText(self.detail_text_show)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ToneMappingApp()
    ex.show()
    sys.exit(app.exec_())