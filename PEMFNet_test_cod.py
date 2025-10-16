import time  # 导入time模块，用于计时或者控制程序执行的时间间隔

import torch  # 导入PyTorch库，这是一个开源的深度学习框架

import torch.nn.functional as F  # 导入PyTorch中的神经网络功能模块，这里通常包含了多种神经网络层的操作函数

import sys  # 导入sys模块，这个模块提供了对Python解释器使用或维护的一些变量的访问，以及与解释器强烈交互的功能

sys.path.append('./models')  # 将'./models'这个路径添加到sys.path中，这样Python就能在这个路径下查找模块了

import numpy as np  # 导入numpy库，numpy是Python的一个强大的数值计算扩展程序库

import os, argparse  # 导入os和argparse模块，os提供了与操作系统交互的功能，argparse用于从命令行解析参数

import cv2  # 导入OpenCV库，这是一个开源的计算机视觉库

# 从'./models'路径下的PRNet模块中导入PRNet类，PRNet可能是一个神经网络模型，用于处理人脸相关的任务
from PEMFNet import Net

# 从当前目录下的data_cod模块中导入test_dataset，这可能是用于测试的数据集加载或处理的类/函数
from data_cod import test_dataset

# 创建一个argparse对象，用于从命令行解析参数
parser = argparse.ArgumentParser()

# 添加一个命令行参数：'--testsize'，类型为整数，默认值为384，帮助信息为'testing size'
# 这个参数可能用于指定测试图像的大小
parser.add_argument('--testsize', type=int, default=384, help='testing size')

# 添加一个命令行参数：'--gpu_id'，类型为字符串，默认值为'0'，帮助信息为'select gpu id'
# 这个参数用于指定使用哪块GPU进行运算
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')

# 添加一个命令行参数：'--test_path'，类型为字符串，默认值为'./COD_datasets/TestDataset/'，帮助信息为'test dataset path'
# 这个参数用于指定测试数据集所在的路径
parser.add_argument('--test_path', type=str, default='./CodDataset/TestDataset/', help='test dataset path')

# 解析命令行参数，返回一个Namespace对象，其中包含所有已解析的参数
opt = parser.parse_args()

# 从解析得到的参数中提取测试数据集路径
dataset_path = opt.test_path

# 设置用于测试的GPU设备
# 根据opt.gpu_id的值来设置环境变量CUDA_VISIBLE_DEVICES，这样PyTorch就只会看到指定的GPU
# 注意：这里只处理了'0'和'1'两种情况，其他值可能会引发错误
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

# 加载模型
# 实例化PEMFNet模型
model = Net()

# 从'./out/PEMFNet_epoch_best.pth'文件中加载模型权重
model.load_state_dict(torch.load('./out/PEMFNet_epoch_best.pth'))

# 将模型移动到GPU上
model.cuda()

# 将模型设置为评估模式，这通常意味着某些层（如Dropout和BatchNorm）在测试时将不会改变其行为
model.eval()

# 定义测试数据集列表
test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']

# 遍历测试数据集列表
for dataset in test_datasets:
    # 构建保存测试结果的路径
    save_path = './test_maps/PEMFNet/' + dataset + '/'

    # 如果保存路径不存在，则创建该路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        # 设置图像和标签的根路径
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'

    # 加载测试数据集
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    # 初始化总时间和计数
    total_time = 0
    count = 0

    # 遍历测试数据集中的每个样本
    for i in range(test_loader.size):
        # 加载图像、标签、名称和用于后处理的图像
        image, gt, name, image_for_post = test_loader.load_data()

        # 将标签转换为float32类型
        gt = np.asarray(gt, np.float32)

        # 归一化标签到[0, 1]范围
        gt /= (gt.max() + 1e-8)

        # 将图像移动到GPU上
        image = image.cuda()

        # 记录开始时间
        start_time = time.perf_counter()

        # 使用模型进行预测，得到多个输出结果
        # res1, _ = model(image)
        res1, res2, res3, res4, res5 = model(image)
        # 将图像输入模型，获取模型的多个输出（可能是模型的不同层或分支的输出）

        # res = res1 + res2 + res3 + res4
        res = res1
        # 记录结束时间
        end_time = time.perf_counter()

        # 更新计数和总时间
        count += 1
        total_time += end_time - start_time

        # 对输出结果进行上采样，使其与标签具有相同的尺寸
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)

        # 应用sigmoid函数，将数据范围限制在[0, 1]之间，并转换为numpy数组
        res = res.sigmoid().data.cpu().numpy().squeeze()

        # 对输出结果进行归一化
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # 保存测试结果的图像
        cv2.imwrite(save_path + name, res * 255)

    # 计算并打印每秒帧数（FPS）
    fps = count / total_time
    print('FPS:', fps)
    # 打印测试完成的消息
    print('Test Done!')

