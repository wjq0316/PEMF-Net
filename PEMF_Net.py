# 导入PyTorch库，PyTorch是一个开源的机器学习库，广泛用于深度学习
import math

import torch

# 从PyTorch库中导入nn模块，这个模块包含了构建神经网络所需的类和函数
import torch.nn as nn

import numpy as np

# 从PyTorch库中导入functional模块，这个模块提供了一些函数式接口，用于实现神经网络中的激活函数、损失函数等
import torch.nn.functional as F

from thop import profile  # 从thop库中导入profile函数，thop是一个用于计算PyTorch模型FLOPs和参数的库

from models.smt import smt_t

from torch import Tensor  # 从torch库中导入Tensor，Tensor是PyTorch中用于存储数据的基本数据结构

from einops import rearrange

from timm.models.layers import trunc_normal_  # 导入timm库中的trunc_normal_函数


################################################################ Edge_Modual #############################################################################
class Edge_Module(nn.Module):

    def __init__(self, in_fea=[64, 512], mid_fea=32):
        super(Edge_Module, self).__init__()

        self.conv1_down = BasicConv2d(in_fea[0], mid_fea, kernel_size=1, padding=0)
        self.conv4_down = BasicConv2d(in_fea[1], mid_fea, kernel_size=1, padding=0)

        self.conv3_1 = BasicConv2d(mid_fea, mid_fea, kernel_size=3, padding=1)
        self.conv3_4 = BasicConv2d(mid_fea, mid_fea, kernel_size=3, padding=1)

        self.concat3 = BasicConv2d(mid_fea * 2, mid_fea * 2, kernel_size=3, padding=1)
        self.concat1 = BasicConv2d(mid_fea * 2, mid_fea * 2, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.conv_down1 = BasicConv2d(mid_fea * 2, mid_fea // 16, kernel_size=1, padding=0)
        self.conv_up1 = BasicConv2d(mid_fea // 16, mid_fea * 2, kernel_size=1, padding=0)

        self.conv_down2 = BasicConv2d(mid_fea * 2, mid_fea // 16, kernel_size=1, padding=0)
        self.conv_up2 = BasicConv2d(mid_fea // 16, mid_fea * 2, kernel_size=1, padding=0)

        self.out = BasicConv2d(mid_fea * 2, 1, kernel_size=3, padding=1)

    def forward(self, x1, x4):
        _, _, h, w = x1.size()
        edge1_fea = self.conv1_down(x1)
        edge1 = self.conv3_1(edge1_fea)

        edge4_fea = self.conv4_down(x4)
        edge4 = self.conv3_4(edge4_fea)

        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)
        edge_concat1 = torch.cat([edge1, edge4], dim=1)
        edge_concat = self.concat1(self.concat3(edge_concat1))

        edge_avg = self.avgpool(edge_concat)
        edge_max = self.maxpool(edge_concat)

        edge_avg = self.conv_down1(edge_avg)
        edge_avg = self.conv_up1(edge_avg)

        edge_max = self.conv_down2(edge_max)
        edge_max = self.conv_up2(edge_max)

        edge_weight = torch.sigmoid(edge_avg + edge_max)

        edge = edge_weight * edge_concat + edge_concat1

        out = self.out(edge)

        return out


################################################################ LGFE 局部-全局特征增强模块 ############################################################

# 定义了一个名为ChannelAttention的PyTorch模块，它实现了一个通道注意力（Channel Attention）层
# 定义一个名为ChannelAttention的类，它继承自nn.Module
class ChannelAttention(nn.Module):
    # 定义一个构造函数__init__，它接受两个参数：输入通道数in_planes和注意力模块的缩减比例ratio（默认为16）
    def __init__(self, in_planes, ratio=16):
        # 调用父类nn.Module的构造函数
        super(ChannelAttention, self).__init__()
        # 初始化一个全局平均池化层，用于计算输入特征的平均值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 初始化一个全局最大池化层，用于计算输入特征的最大值
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 初始化第一个卷积层，用于计算平均池化后的特征
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        # 初始化ReLU激活函数
        self.relu1 = nn.ReLU()
        # 初始化第二个卷积层，用于计算最大池化后的特征
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        # 初始化Sigmoid函数，用于生成最终的通道注意力权重
        self.sigmoid = nn.Sigmoid()

    # 定义一个前向传播方法forward，它接受一个输入x
    def forward(self, x):
        # 应用第一个卷积层和ReLU激活函数，然后应用全局平均池化层，得到平均池化后的特征
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 应用第一个卷积层和ReLU激活函数，然后应用全局最大池化层，得到最大池化后的特征
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 将平均池化后的特征和最大池化后的特征相加
        out = avg_out + max_out
        # 应用Sigmoid函数得到最终的通道注意力权重
        return self.sigmoid(out)


# 定义了一个名为SpatialAttention的PyTorch模块，它实现了一个空间注意力（Spatial Attention）层
# 定义一个名为SpatialAttention的类，它继承自nn.Module
class SpatialAttention(nn.Module):
    # 定义一个构造函数__init__，它接受一个参数kernel_size，表示卷积核的大小，默认为7
    def __init__(self, kernel_size=7):
        # 调用父类nn.Module的构造函数
        super(SpatialAttention, self).__init__()
        # 检查kernel_size是否为3或7，如果不是，则抛出异常
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 计算padding值，用于卷积层的padding参数
        padding = 3 if kernel_size == 7 else 1

        # 初始化卷积层，用于计算空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # 初始化Sigmoid函数，用于生成最终的注意力权重
        self.sigmoid = nn.Sigmoid()

    # 定义一个前向传播方法forward，它接受一个输入x
    def forward(self, x):
        # 计算平均池化后的特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 计算最大池化后的特征
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均池化后的特征和最大池化后的特征拼接在一起
        x = torch.cat([avg_out, max_out], dim=1)
        # 应用卷积层处理拼接后的特征
        x = self.conv1(x)
        # 应用Sigmoid函数生成最终的注意力权重
        return self.sigmoid(x)


################################################################ CFM 特征融合模块 ################################################################
# 这段代码定义了一个名为BasicConv2d的PyTorch模块，它实现了一个基础的二维卷积块
# 定义一个名为BasicConv2d的类，它继承自nn.Module
class BasicConv2d(nn.Module):
    # 定义一个构造函数__init__，它接受六个参数：输入通道数in_planes、输出通道数out_planes、卷积核大小kernel_size、步长stride（默认为1）、填充padding（默认为0）和膨胀dilation（默认为1）
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        # 调用父类nn.Module的构造函数
        super(BasicConv2d, self).__init__()
        # 初始化一个卷积层，使用指定的参数。nn.Conv2d是PyTorch中的一个函数，用于创建二维卷积层
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        # 初始化一个批量归一化层，使用指定的输出通道数
        self.bn = nn.BatchNorm2d(out_planes)
        # 初始化一个ReLU激活函数，使用in-place模式
        self.relu = nn.ReLU(inplace=True)

    # 定义一个前向传播方法forward，它接受一个输入x
    def forward(self, x):
        # 应用卷积层处理输入x
        x = self.conv(x)
        # 应用批量归一化层处理卷积层的输出
        x = self.bn(x)

        x = self.relu(x)

        # 返回处理后的特征x
        return x


class DWPWConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),  # 3x3深度卷积层
            nn.BatchNorm2d(inc),  # 批归一化层
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),  # 1x1卷积层
            nn.BatchNorm2d(outc),  # 批归一化层
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)  # 通过卷积层并返回输出


#####################################################  MSFF #########################################################################

class CAEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CAEM, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=1, dilation=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1)

        self.sa = SpatialAttention()

        self.fusion = BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv_down = BasicConv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)

        x1 = self.branch1(x)

        x2 = self.branch2(x)

        x3 = self.branch3(x)

        x_fused = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x_sa = self.sa(x_fused) * x_fused

        x_ff = x + x_sa

        x_out = self.fusion(x_ff)

        out = self.conv_down(x_out)

        return out


# 定义PyramidPooling类的构造函数，它接受输入通道数in_channel和输出通道数out_channel作为参数   AGE
class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 调用基类的构造函数
        super(PyramidPooling, self).__init__()

        # 计算隐藏层的通道数，为输入通道数的1/4
        hidden_channel = int(in_channel / 4)

        # 定义四个卷积层，它们具有相同的内核大小k=1，步长s=1，填充p=0
        self.conv1 = BasicConv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = BasicConv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.conv3 = BasicConv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.conv4 = BasicConv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)

        # 定义一个输出卷积层，其输入通道数为in_channel * 2（原始通道加上四个池化后的通道），输出通道数为out_channel
        self.out = BasicConv2d(in_channel * 2, out_channel, kernel_size=1, stride=1, padding=0)

    # 定义前向传播函数
    def forward(self, x):
        # 获取输入x的空间维度大小
        size = x.size()[2:]

        # 对于任何给定的输入尺寸，F.adaptive_avg_pool2d(x, k) 将输出特征图的高度和宽度都调整为 k 个像素点
        # 对x进行1x1平均池化，然后使用上采样恢复到原始大小，并应用第一个卷积层
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)

        # 对x进行2x2平均池化，然后使用上采样恢复到原始大小，并应用第二个卷积层
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)

        # 对x进行3x3平均池化，然后使用上采样恢复到原始大小，并应用第三个卷积层
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)

        # 对x进行6x6平均池化，然后使用上采样恢复到原始大小，并应用第四个卷积层
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)

        # 将原始输入x与四个经过池化和卷积的特征图沿通道维度拼接起来
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)

        # 将拼接后的特征图通过输出卷积层
        x = self.out(x)

        # 返回最终的输出
        return x


class EGFM(nn.Module):
    def __init__(self, channel, rates=[1, 6, 12, 18]):
        super(EGFM, self).__init__()

        self.branch0 = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3,
                      dilation=rates[0], padding=rates[0], bias=False),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True)
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3,
                      dilation=rates[1], padding=rates[1], bias=False),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3,
                      dilation=rates[2], padding=rates[2], bias=False),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(channel // 4, channel // 4, kernel_size=3,
                      dilation=rates[3], padding=rates[3], bias=False),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True)
        )

        self.cat = BasicConv2d(channel, channel, kernel_size=3, padding=1)

        self.context = PyramidPooling(channel, channel)

        self.end = BasicConv2d(channel, channel, kernel_size=3, padding=1)

        self.channel = channel

    def forward(self, x, edge):
        edge_small = F.interpolate(edge, size=x.size()[2:], mode='bilinear', align_corners=False)

        fused_edge_input = x * edge_small + x

        fused_c = list(torch.chunk(fused_edge_input, 4, dim=1))

        branch0 = self.branch0(fused_c[0])
        branch1 = self.branch1(fused_c[1])
        branch2 = self.branch2(fused_c[2])
        branch3 = self.branch3(fused_c[3])

        branch = torch.cat((branch0, branch1, branch2, branch3), dim=1)
        branch_cat = self.cat(branch)


        branch_ca_up, _ = torch.split(branch_cat, [self.channel // 2, self.channel // 2], dim=1)

        context = self.context(x)

        context_ca_up, _ = torch.split(context, [self.channel // 2, self.channel // 2], dim=1)

        out = self.end(torch.cat((branch_ca_up, context_ca_up), dim=1))

        return out


class Decoder(nn.Module):
    def __init__(self, channel=32):
        super(Decoder, self).__init__()
        self.predict_layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Conv2d(channel, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        prediction = self.predict_layer(x)

        return prediction


################################################ Net #################################################################################################
# 定义Net类，这是一个深度学习网络模型
class Net(nn.Module):
    def __init__(self):
        # 调用基类的构造函数
        super(Net, self).__init__()

        # 构建主干网络
        self.smt = smt_t()

        # 构建边缘特征模块
        self.edge_layer = Edge_Module(in_fea=[64, 512])

        self.Trans1 = BasicConv2d(256, 256, kernel_size=3, padding=1)
        self.Trans2 = BasicConv2d(128, 128, kernel_size=3, padding=1)
        self.Trans3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.decoder = nn.ModuleList([
            Decoder(channel=32),
            Decoder(channel=64),
            Decoder(channel=128),
            Decoder(channel=256),
        ])

        # 创建一个包含RFB_modified模块的列表，用于特征的增强处理
        self.caem = nn.ModuleList([
            CAEM(64, 32),
            CAEM(128, 64),
            CAEM(256, 128),
            CAEM(512, 256)
        ])

        self.egfm = nn.ModuleList([
            EGFM(64, [1, 2, 3, 4]),
            EGFM(128, [1, 2, 3, 4]),
            EGFM(256, [1, 2, 3, 4]),
            EGFM(512, [1, 2, 3, 4])
        ])

    # 定义前向传播函数  x.shape = [1,3,320,320]
    # 将输入x通过模型进行前向传播，shape参数用于指定输出尺寸，epoch参数可能用于指定训练的当前周期，但在本函数中未使用
    def forward(self, x, shape=None):
        ############################################### 获取最初RGB输入x的空间分辨率大小#################################

        # 如果没有提供shape，则使用输入x的空间尺寸（高度和宽度） [384,384]
        shape = x.size()[2:] if shape is None else shape
        ################################################ 获取各阶段特征 ###############################################
        # 通过主干网络提取不同阶段的特征，假设self.bkbone()返回一个包含4个阶段特征的元组
        # torch.Size([1, 64, 96, 96]) torch.Size([1, 128, 48, 48]) torch.Size([1, 256, 24, 24]) torch.Size([1, 512, 12, 12])
        bk_stage1, bk_stage2, bk_stage3, bk_stage4 = self.smt(x)

        ################################################ 获取边缘特征 #################################################
        edge_map = self.edge_layer(bk_stage1, bk_stage4)

        # 使用插值方法放大edge_map特征的尺寸  [1,1,384,384]
        edge_prediction = F.interpolate(edge_map, size=(384,384), mode='bilinear', align_corners=False)
        ################################################ MAFM ####################################

        bk_edge4 = self.egfm[3](bk_stage4, edge_map)  # [1,512,12,12]
        bk_edge3 = self.egfm[2](bk_stage3, edge_map)  # [1,256,24,24]
        bk_edge2 = self.egfm[1](bk_stage2, edge_map)  # [1,128,48,48]
        bk_edge1 = self.egfm[0](bk_stage1, edge_map)  # [1,64,96,96]

        ################################################ CAEM ####################################

        out4 = self.caem[3](bk_edge4)
        prediction4 = F.interpolate(self.decoder[3](out4), size=shape, mode='bilinear', align_corners=False)

        out4_up = F.interpolate(out4, bk_edge3.size()[2:], mode='bilinear', align_corners=False)
        out4_up, _ = torch.split(out4_up, [128, 128], dim=1)
        bk_edge3_up, _ = torch.split(bk_edge3, [128, 128], dim=1)
        out3 = torch.cat((out4_up, bk_edge3_up), dim=1)
        out3 = self.Trans1(out3)
        out3 = self.caem[2](out3)
        prediction3 = F.interpolate(self.decoder[2](out3), size=shape, mode='bilinear', align_corners=False)

        out3_up = F.interpolate(out3, bk_edge2.size()[2:], mode='bilinear', align_corners=False)
        out3_up, _ = torch.split(out3_up, [64, 64], dim=1)
        bk_edge2_up, _ = torch.split(bk_edge2, [64, 64], dim=1)
        out2 = torch.cat((out3_up, bk_edge2_up), dim=1)
        out2 = self.Trans2(out2)
        out2 = self.caem[1](out2)
        prediction2 = F.interpolate(self.decoder[1](out2), size=shape, mode='bilinear', align_corners=False)

        out2_up = F.interpolate(out2, bk_edge1.size()[2:], mode='bilinear', align_corners=False)
        out2_up, _ = torch.split(out2_up, [32, 32], dim=1)
        bk_edge1_up, _ = torch.split(bk_edge1, [32, 32], dim=1)
        out1 = torch.cat((out2_up, bk_edge1_up), dim=1)
        out1 = self.Trans3(out1)
        out1 = self.caem[0](out1)
        prediction1 = F.interpolate(self.decoder[0](out1), size=shape, mode='bilinear', align_corners=False)

        return prediction1, prediction2, prediction3, prediction4, edge_prediction

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)['model'])


if __name__ == '__main__':
    # 创建Net类的实例，传递配置参数
    net = Net()

    # 创建一个形状为 (1, 3, 384, 384) 的随机张量
    x = torch.randn(1, 3, 384, 384)

    # 使用Net类的实例和输入张量x来计算FLOPs和参数数量
    flops, params = profile(net, (x,))

    # 打印FLOPs和参数数量
    print('flops: %.2f G, parms: %.2f M' % (flops / 1000000000.0, params / 1000000.0))


