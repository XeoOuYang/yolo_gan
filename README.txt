建立conda环境，下载安装Anconda（https://www.anaconda.com/）
conda create --name YOLO_v8 python=3.10

激活conda环境，安装依赖
conda activate YOLO_v8

安装yolo_v8（https://github.com/ultralytics/ultralytics）
pip install ultralytics

测试安装环境
Python main.py

测试cuda环境
check_cuda_env()
##### 输出结果如下：注意torch版本需要gpu，这样才能使用cuda的能力
Ultralytics YOLOv8.2.35 🚀 Python-3.10.14 torch-2.3.1+cpu CPU (Intel Core(TM) i9-14900KF)
Setup complete ✅ (32 CPUs, 63.8 GB RAM, 183.2/803.2 GB disk)
# 安装方法如下
pip install torch==2.3.1+cu121 torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu121

labelImg安装使用
https://github.com/HumanSignal/labelImg

yolov8模型结构说明
https://developer.aliyun.com/article/1430611

yolov8改进
1、小目标检查算法：《Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection》
https://arxiv.org/abs/2202.06934
无人机、高空拍摄
https://cloud.tencent.com/developer/article/2419080
https://cloud.tencent.com/developer/article/2331361
2、点采样：《Learning to Upsample by Learning to Sample》
https://arxiv.org/abs/2308.15085
3、CAFM注意力机制：《Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising》
https://arxiv.org/pdf/2403.10067

看着两个就行了：
https://github.com/iscyy/ultralyticsPro
https://github.com/z1069614715/objectdetection_script
https://github.com/phd-benel/yolov8_improved_exp
https://github.com/iscyy/yoloair

其他资料：（坑钱的，当作扩展了解关键词）
https://blog.csdn.net/qq_37706472/article/details/129352058
https://blog.csdn.net/m0_67647321/article/details/139703508?spm=1001.2014.3001.5502
https://www.bilibili.com/video/BV1Gj411D7Pf/?vd_source=f21c2001f0b9b7072aadef4fe02c0398

注意力机制：
1、《CBAM: Convolutional Block Attention Module》
https://arxiv.org/abs/1807.06521
2、《ECA-Net：Efficient Channel Attention for Deep Convolutional Neural Networks》
https://arxiv.org/pdf/1910.03151.pdf
3、《Coordinate Attention for Efficient Mobile Network Design》
https://arxiv.org/pdf/2103.02907.pdf
4、MHSA
参考：https://developer.aliyun.com/article/1462155
（可以在阿里云里面广泛搜索一下，基本上都有代码和步骤，不像csdn这么坑钱）

YOLO综述
https://developer.aliyun.com/article/1508518
https://cloud.tencent.com/developer/article/2406045

YOLO热力图
https://github.com/z1069614715/objectdetection_script/tree/master/yolo-gradcam

https://github.com/phd-benel/yolov8_improved_exp