# BTKD
# 环境：
## 我们的环境是python==3.9, numpy==1.21.0, pytorch==1.9.0, cuda==11.3, cudnn==8.0,opencv-python==4.1.2.30, matplotlib==3.4.3,都是一些基本库。
## 如要运行yolo模型，需要安装ultralytics库
## 如要运行mit分割模型，需要安装timm相关库
## Ubuntu和Windows系统都能运行，就是文件路径的写法不一样
## 分类的预训练权重下载好放在主目录的pth文件夹中
## 分割的预训练权重和训练好的权重下载好放在segment文件夹下的pth文件夹中
## 我的显卡是RTX4090*2
# 运行：
## 分类
## dataload_cf100.py文件里面改你的数据集的路径
## train_XXX_YYY名字的是直接训练的代码，XXX是模型选择，YYY是数据集选择
## dis_开头的是知识蒸馏的代码，运行时先修改读取pth文件的路径
## 目标检测
## 在detect文件夹中运行train开头的代码
## 分割
## 运行segment文件夹下的train开头的代码
## 预训练权重：
## 教师：deeplabv3_resnet101.pth  ||   学生：deeplabv3_resnet18.pth  ||  best_model：btkd_deeplabv3_resnet18.pth
## 教师：mit_B4.pth  ||   学生：mit_B0.pth  ||  best_model：btkd_mit_B0.pth
# 补充：
## 能力和精力都有限，后续会慢慢更新代码


# To Be Continue...
