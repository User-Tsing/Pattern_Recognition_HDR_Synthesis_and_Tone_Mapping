Directed by STAssn     

模式识别：基于数字图像处理的色调映射与HDR图像合成，Python语言编写      

色调映射部分：传统方法OpenCV： Reinhard色调映射可变参数，ACES等，均有指标展示     
HDR合成部分：深度学习方法生成对抗网络WGAN-GP，生成器采用UNet结构加残差连接，判别器采用CNN含谱归一化的卷积神经网络结构，未启用注意力机制       
运行方法：PyCharm配置环境，然后运行main.py      
训练模型：源码采用VDS数据集，可自己寻找其他数据集然后修改data_load_gan.py适配导入模板           


Directed by STAssn      
Participants: F Lu, C Wang
