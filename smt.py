import torch  # 导入PyTorch库，PyTorch是一个开源的深度学习库

import torch.nn as nn  # 导入PyTorch的神经网络模块，它包含构建神经网络所需的各种层

import torch.nn.functional as F  # 导入PyTorch的神经网络功能函数模块，包含激活函数、损失函数等

from functools import partial  # 导入functools库中的partial函数，用于固定函数的部分参数

# 从timm库中导入一些模块和函数
from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # 导入timm库中的层相关函数和类
from timm.models.registry import register_model  # 导入timm库中的模型注册函数
from timm.models.vision_transformer import _cfg  # 导入timm库中vision_transformer的配置

import math  # 导入math库，用于数学运算

# 从torchvision库中导入图像预处理相关的模块和函数
from torchvision import transforms  # 导入torchvision库中的图像预处理模块
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # 导入timm库中定义的ImageNet数据集的默认均值和标准差
from timm.data import create_transform  # 导入timm库中创建图像预处理转换的函数
from timm.data.transforms import str_to_pil_interp  # 导入timm库中字符串到PIL插值方法的转换函数

# 从ptflops库中导入获取模型复杂度的函数
from ptflops import get_model_complexity_info  # 导入ptflops库中的函数，用于获取模型的FLOPs和参数数量

# 从thop库中导入模型性能分析函数
from thop import profile  # 导入thop库中的profile函数，用于分析模型的FLOPs和参数


# 定义一个名为Mlp的类，继承自PyTorch的基础神经网络模块nn.Module
class Mlp(nn.Module):

    # 初始化函数，接受输入特征数、隐藏层特征数（默认为输入特征数）、输出特征数（默认为输入特征数）、激活函数层（默认为GELU）和Dropout概率
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):

        # 调用父类nn.Module的初始化函数
        super().__init__()

        # 如果out_features未指定，则使用in_features作为默认值
        out_features = out_features or in_features

        # 如果hidden_features未指定，则使用in_features作为默认值
        hidden_features = hidden_features or in_features

        # 定义第一个全连接层，将输入特征映射到隐藏层特征
        self.fc1 = nn.Linear(in_features, hidden_features)

        # 定义深度可分离卷积层，此处假设DWConv是一个已定义好的类
        self.dwconv = DWConv(hidden_features)
        '''
        深度可分离卷积层（Depthwise Separable Convolution Layer）是卷积神经网络中对标准的卷积计算进行改进所得到的层结构。其核心思想是将标准的卷积操作分解为两个独立的步骤：深度卷积（Depthwise Convolution）
        和逐点卷积（Pointwise Convolution）。深度卷积在每个输入通道上独立进行空间卷积操作，即对通道（深度）分别进行空间卷积，并将输出进行拼接。而逐点卷积则使用单位卷积核（如1×1卷积）对深度卷积的输出进行通道卷积，
        以得到特征图。这种分离的方式可以大大减少参数量和计算量，同时提升卷积核参数的使用效率。深度可分离卷积层在应用中具有诸多优势。首先，由于其参数量的减少，模型复杂度降低，有助于防止过拟合
        其次，计算量的减少使得它在移动设备等计算资源受限的场景下具有更好的性能。因此，深度可分离卷积层被广泛用于微型神经网络的搭建以及大规模卷积神经网络的结构优化中
        '''
        # 实例化激活函数层
        self.act = act_layer()

        # 定义第二个全连接层，将隐藏层特征映射到输出特征
        self.fc2 = nn.Linear(hidden_features, out_features)

        # 定义Dropout层，用于防止过拟合
        self.drop = nn.Dropout(drop)

        # 应用权重初始化函数到模型的所有层
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 定义权重初始化函数，m代表模型中的某一层

        if isinstance(m, nn.Linear):
            # 如果m是一个全连接层
            trunc_normal_(m.weight, std=.02)
            # 使用截断正态分布初始化全连接层的权重，标准差为0.02

            if isinstance(m, nn.Linear) and m.bias is not None:
                # 再次检查m是否为全连接层且该层有偏置项
                nn.init.constant_(m.bias, 0)
                # 将全连接层的偏置项初始化为0

        elif isinstance(m, nn.LayerNorm):
            # 如果m是一个LayerNorm层
            nn.init.constant_(m.bias, 0)
            # 将LayerNorm层的偏置项初始化为0
            nn.init.constant_(m.weight, 1.0)
            # 将LayerNorm层的权重初始化为1.0

        elif isinstance(m, nn.Conv2d):
            # 如果m是一个二维卷积层
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 计算输出特征图的元素个数，即卷积核大小乘以输出通道数

            fan_out //= m.groups
            # 考虑分组卷积的情况，将输出特征图的元素个数除以分组数

            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 使用正态分布初始化卷积层的权重，方差为2/fan_out

            if m.bias is not None:
                # 如果卷积层有偏置项
                m.bias.data.zero_()
                # 将卷积层的偏置项初始化为0

    def forward(self, x, H, W):
        # 定义前向传播方法，输入包括数据x、高度H和宽度W

        x = self.fc1(x)
        # 通过第一个全连接层（fc1）进行线性变换，将输入x映射到新的特征空间

        x = self.act(x + self.dwconv(x, H, W))
        # 调用深度可分离卷积层（dwconv）对x进行卷积操作，输入还包括高度H和宽度W
        # 将深度可分离卷积的输出与原始输入x相加（可能是一个残差连接）
        # 然后，通过激活函数（act）对相加后的结果进行非线性变换

        x = self.drop(x)
        # 应用dropout层（drop），随机将x中的部分元素置零，以防止过拟合

        x = self.fc2(x)
        # 通过第二个全连接层（fc2）进行线性变换，进一步映射特征

        x = self.drop(x)
        # 再次应用dropout层，进一步防止过拟合

        return x
        # 返回前向传播的结果x


'''
第一stage时调用Attention类，传入dim=64,ca_num_heads=ca_num_heads=4, sa_num_heads=sa_num_heads=-1, qkv_bias=qkv_bias=True, qk_scale=qk_scale=Nonne,attn_drop=attn_drop=0, 
proj_drop=drop=0, ca_attention=ca_attention=1（使用通道注意力，而是使用普通的空间注意力）,expand_ratio=expand_ratio=2,  初始化函数会计算通道数除以通道头数是否可以整除，然后将结果传给分组卷积的组数。 
然后接着判断，因为这里ca_attention=1，所以使用通道注意力，第一个线性层(启用偏置)表示通过输入X,生成Value，第二个线性层(启用偏置)生成s,然后生成卷积组 这里是4个头，产生4个分组卷积，输入和输出通道数均是64/4=16 
第一个卷积层：nn.Conv2d(16, 16, kernel_size=3 ,padding=1, stride=1,groups=16)  第二个卷积层：nn.Conv2d(16, 16, kernel_size=5 ,padding=2, stride=1,groups=16)  
第三个卷积层：nn.Conv2d(16, 16, kernel_size=7 ,padding=3, stride=1,groups=16)  第四个卷积层：nn.Conv2d(16, 16, kernel_size=9 ,padding=4, stride=1,groups=16)  

然后初始化一个1x1卷积层，用于扩展特征维度nn.Conv2d(64, dim * expand_ratio=128, kernel_size=1, padding=0, stride=1,groups=16)  
然后产生一个批归一化层nn.BatchNorm2d(dim*expand_ratio=128)  
初始化另一个1x1卷积层，用于恢复原始维度nn.Conv2d(dim * expand_ratio=128, dim=64, kernel_size=1, padding=0, stride=1)  

这里传入的是通过层归一化后的输入，输入形状是[4,56*56,64], H=W=56，C=64,前向传播函数中先获取输入的B=4,N=56*56,C=64 然后使用的是通道注意力进入if循环，先在左边那条路线生成V，大小依然是[4,56*56,64]，然后通过右边先线性层生成S，大小形状
不会改变，依然是[4,56*56,64]，然后重塑S[4,56,56,4,16],在换维得到形状[4，4，16，56，56] 第一维表示头数，第二维批量大小，第三维通道数，第四维高，第五维宽。  

然后进入多头混合卷积阶段，先依次获取每一个分组卷积，通过s[i]先获取每一个头的输入，也就是分别获取4个头的输入,每个输入是[4,16,56,56]. 然后将输入传入每一个分组卷积中，最终的每一个分组卷积的输出结果为  
[4,16,(56-3+2*1+1)/1,(56-3+2*1+1)/1]=[4,16,56,56]   [4,16,(56-5+2*2+1)/1,(56-5+2*2+1)/1]=[4,16,56,56]  
[4,16,(56-7+2*3+1)/1,(56-7+2*3+1)/1]=[4,16,56,56]   [4,16,(56-9+2*4+1)/1,(56-9+2*4+1)/1]=[4,16,56,56]  
然后在对每一个卷积输出的结果进行重塑都是这样的形状：[4,16,1,56,56]  
然后将第一个输入[4,16,1,56,56]传入给s_out  然后将第二个，第三个，第四个，在第二维依次拼接到第一个输入上[4,16,1+1+1+1,56,56]=[4,16,4,56,56]  

最终分组卷积结束后的结果在重塑为[4,64,56,56]  

然后通过第一个1*1的卷积层 输出为[4,128,(56-1+2*0+1)/1,(56-1+2*0+1)/1]=[4,128,56,56]  
然后通过批归一化，在通过激活函数后输入形状不变仍然为[4,128,56,56]  
然后通过第二个1*1的卷积层 输出为[4,64,(56-1+2*0+1)/1,(56-1+2*0+1)/1]=[4,64,56,56]  

然后将结果传入尺度聚合属性中保存结果  
然后将其重塑为[4,64,56*56] 在换维为[4,56*56,64]  
最后和V[4,56*56,64]对位相乘得到最终的SAM结果 依然为[4,56*56,64]  然后就进行两次残差链接，注意第一、第二stage不适用MSA  

=================================================================================================================================================================================

注意这里使用了两个block，所以第一个block输出之后，会传入第二个block中，再次传入参数dim=64,ca_num_heads=ca_num_heads=4, sa_num_heads=sa_num_heads=-1, qkv_bias=qkv_bias=True, qk_scale=qk_scale=Nonne,
attn_drop=attn_drop=0, proj_drop=drop=0, ca_attention=ca_attention=1（使用通道注意力，而是使用普通的空间注意力）,expand_ratio=expand_ratio=2,  初始化函数会计算通道数除以通道头数是否可以整除，
然后将结果传给分组卷积的组数。 然后接着判断，因为这里ca_attention=1，所以使用通道注意力，第一个线性层(启用偏置)表示通过输入X,生成Value，第二个线性层(启用偏置)生成s,然后生成卷积组 这里是4个头，产生4个分组卷积，
输入和输出通道数均是64/4=16 
第一个卷积层：nn.Conv2d(16, 16, kernel_size=3 ,padding=1, stride=1,groups=16)  第二个卷积层：nn.Conv2d(16, 16, kernel_size=5 ,padding=2, stride=1,groups=16)  
第三个卷积层：nn.Conv2d(16, 16, kernel_size=7 ,padding=3, stride=1,groups=16)  第四个卷积层：nn.Conv2d(16, 16, kernel_size=9 ,padding=4, stride=1,groups=16)  

然后初始化一个1x1卷积层，用于扩展特征维度nn.Conv2d(64, dim * expand_ratio=128, kernel_size=1, padding=0, stride=1,groups=16)  
然后产生一个批归一化层nn.BatchNorm2d(dim*expand_ratio=128)  
初始化另一个1x1卷积层，用于恢复原始维度nn.Conv2d(dim * expand_ratio=128, dim=64, kernel_size=1, padding=0, stride=1)  

这里传入的是通过层归一化后的输入，输入形状是[4,56*56,64], H=W=56，C=64,前向传播函数中先获取输入的B=4,N=56*56,C=64 然后使用的是通道注意力进入if循环，先在左边那条路线生成V，大小依然是[4,56*56,64]，然后通过右边先线性层生成S，大小形状
不会改变，依然是[4,56*56,64]，然后重塑S[4,56,56,4,16],在换维得到形状[4，4，16，56，56] 第一维表示头数，第二维批量大小，第三维通道数，第四维高，第五维宽。  

然后进入多头混合卷积阶段，先依次获取每一个分组卷积，通过s[i]先获取每一个头的输入，也就是分别获取4个头的输入,每个输入是[4,16,56,56]. 然后将输入传入每一个分组卷积中，最终的每一个分组卷积的输出结果为  
[4,16,(56-3+2*1+1)/1,(56-3+2*1+1)/1]=[4,16,56,56]   [4,16,(56-5+2*2+1)/1,(56-5+2*2+1)/1]=[4,16,56,56]  
[4,16,(56-7+2*3+1)/1,(56-7+2*3+1)/1]=[4,16,56,56]   [4,16,(56-9+2*4+1)/1,(56-9+2*4+1)/1]=[4,16,56,56]  
然后在对每一个卷积输出的结果进行重塑都是这样的形状：[4,16,1,56,56]  
然后将第一个输入[4,16,1,56,56]传入给s_out  然后将第二个，第三个，第四个，在第二维依次拼接到第一个输入上[4,16,1+1+1+1,56,56]=[4,16,4,56,56]  

最终分组卷积结束后的结果在重塑为[4,64,56,56]  

然后通过第一个1*1的卷积层 输出为[4,128,(56-1+2*0+1)/1,(56-1+2*0+1)/1]=[4,128,56,56]  
然后通过批归一化，在通过激活函数后输入形状不变仍然为[4,128,56,56]  
然后通过第二个1*1的卷积层 输出为[4,64,(56-1+2*0+1)/1,(56-1+2*0+1)/1]=[4,64,56,56]  

然后将结果传入尺度聚合属性中保存结果  
然后将其重塑为[4,64,56*56] 在换维为[4,56*56,64]  
最后和V[4,56*56,64]对位相乘得到最终的SAM结果 依然为[4,56*56,64]  然后就进行两次残差链接，注意第一、第二stage不使用MSA  
**********************************************************************************************************************************************************************************

其中stage1和stage2是SAM Block，他是全卷积结构，主要由MHMC和SAA两个模块构成结果其实是差不多的，先通过Patch Embedding层进行下采样，然后通过SAM进行MHMC和SAA 输出结果为[4,28*28,128]  

然后进入stage3，然后先进行下采样，结果为[4,14*14,256],第一个block依然是SAM尺度聚合模块 然后外加mlp， 第二个block的时候此时传入的参数ca_attention=0,这里不使用通道注意力了，改用空间注意力
dim=256, ca_num_heads=4, sa_num_heads=8, qkv_bias=True, qk_scale=None,attn_drop=0., proj_drop=0., ca_attention=0, expand_ratio=2  首先断言计算dim能否整除ca_num_heads和sa_num_heads
然后这里ca_attention=0，所以进入else语句执行， head_dim = 256//8=32, 缩放因子是 256**-0.5, 然后通过线性层计算查询Q, 然后通过线性层计一起计算K和V，维度变成2倍，然后定义卷积层
Conv2d(256, 256, kernel_size=3, padding=1, stride=1, groups=256)，  然后进入前向传播函数，通过线性层计算查询Q，再将查询Q改变维度[4,14*14,8,32],再换维[4,8,14*14,32]
然后计算kv，输出[4,14*14,512],再改变维度[4,14*14,2,8,64] 然后换维[2,4,8,14*14,32] ,然后通过kv[0],kv[1]，分别获取k和v，形状均是[4,8,14*14,32]
然后计算注意力分数，attn = (q @ k.transpose(-2, -1)) * self.scale   然后在经过softmax在最后一个维度求解，然后通过一个dropout层，然后在和v点乘，得到最终的多头注意力形状为[4,8,14*14,32]
然后转置得[4,14*14,8,32],然后再reshape成[4,14*14,256]。然后再将v转置得[4,14*14,8,32]再reshape[4,14*14,256],再转置[4,256,14*14],再重塑为[4,256,14,14],然后传入分组卷积中
Conv2d(256, 256, kernel_size=3, padding=1, stride=1, groups=256) 得到结果为[4,256,(14-3+2*1+1)/1,(14-3+2*1+1)/1]=[4,256,14,14],然后与多头注意力结果相加输出为[4,256,14,14]
再重塑为[4,256,14*14],再转置为[4,14*14,256]。然后再通过一个线性层为[4,14*14,256],再通过dropout输出第2个block的输出，也是第三个block的输入 

然后block3=SAM block4=MSA block5=SAM block6=MSA block7=SAM block8=MSA

====================================================================================================================================================================================
第四阶段的输入参数为传入dim=512,ca_num_heads=ca_num_heads=-1, sa_num_heads=sa_num_heads=16, qkv_bias=qkv_bias=True, qk_scale=qk_scale=Nonne,attn_drop=attn_drop=0, 
proj_drop=drop=0, ca_attention=ca_attention=1（使用通道注意力，而是使用普通的空间注意力）,expand_ratio=expand_ratio=2,  首先断言计算dim能否整除ca_num_heads和sa_num_heads
然后这里ca_attention=0，所以进入else语句执行， head_dim = 512//16=32, 缩放因子是 512**-0.5, 然后通过线性层计算查询Q, 然后通过线性层计一起计算K和V，维度变成2倍，然后定义卷积层
Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512)，  然后进入前向传播函数，输入x为 [4,7*7,512] 通过线性层计算查询Q，再将查询Q改变维度[4,7*7,16,32],再换维[4,8,7*7,32]
然后计算kv，输出[4,7*7,1024],再改变维度[4,7*7,2,16,32] 然后换维[2,4,16,7*7,32] ,然后通过kv[0],kv[1]，分别获取k和v，形状均是[4,16,7*7,32]
然后计算注意力分数，attn = (q @ k.transpose(-2, -1)) * self.scale   然后在经过softmax在最后一个维度求解，然后通过一个dropout层，然后在和v点乘，得到最终的多头注意力形状为[4,16,7*7,32]
然后转置得[4,7*7,16,32],然后再reshape成[4,7*7,512]。然后再将v转置得[4,7*7,16,32]再reshape[4,7*7,512],再转置[4,512,7*7],再重塑为[4,512,7,7],然后传入分组卷积中
Conv2d(512, 512, kernel_size=3, padding=1, stride=1, groups=512) 得到结果为[4,512,(7-3+2*1+1)/1,(7-3+2*1+1)/1]=[4,512,7,7],然后与多头注意力结果相加输出为[4,512,7,7]
再重塑为[4,512,7*7],再转置为[4,7*7,512]。然后再通过一个线性层为[4,7*7,512],再通过dropout输出,这也是最后一个stage的输出


'''


# 定义一个名为Attention的PyTorch模块
class Attention(nn.Module):

    # 初始化函数，接受多个参数，包括输入维度dim、两种注意力机制的头数、偏置项、缩放因子、dropout率以及扩展比率
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):

        # 调用父类nn.Module的初始化方法
        super().__init__()

        # 设置是否使用通道注意力（CA） 只是在第三个stage中的前三个MSA才使用通道注意力
        self.ca_attention = ca_attention

        # 设置输入维度
        self.dim = dim

        # 设置通道注意力的头数
        self.ca_num_heads = ca_num_heads

        # 设置空间注意力的头数
        self.sa_num_heads = sa_num_heads

        # 断言确保输入维度能被通道注意力的头数整除
        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."

        # 断言确保输入维度能被空间注意力的头数整除
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        # 初始化激活函数为GELU
        self.act = nn.GELU()

        # 初始化一个线性层，用于投影或转换特征维度
        self.proj = nn.Linear(dim, dim)

        # 初始化一个dropout层，用于在投影后防止过拟合
        self.proj_drop = nn.Dropout(proj_drop)

        # 计算分组卷积的组数，用于通道注意力
        self.split_groups = self.dim // ca_num_heads

        # 如果使用通道注意力
        if ca_attention == 1:

            # 初始化线性层v，用于转换值向量(Value)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)

            # 初始化线性层s，可能是用于某种转换或缩放操作
            self.s = nn.Linear(dim, dim, bias=qkv_bias)

            # 对于每个通道注意力的头
            for i in range(self.ca_num_heads):
                # 初始化一个分组卷积层，每个头的卷积核大小以此增加2，和填充以此增加1，步幅不变，不同
                local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                       padding=(1 + i), stride=1, groups=dim // self.ca_num_heads)

                # 将卷积层添加到当前对象的属性中
                setattr(self, f"local_conv_{i + 1}", local_conv)

            # 初始化一个1x1卷积层，用于扩展特征维度
            self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                                   groups=self.split_groups)

            # 初始化批量归一化层
            self.bn = nn.BatchNorm2d(dim * expand_ratio)

            # 初始化另一个1x1卷积层，用于恢复原始维度
            self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        # 如果不使用通道注意力
        else:
            # 计算每个空间注意力头的维度
            head_dim = dim // sa_num_heads

            # 初始化缩放因子，用于注意力分数的缩放，在这里都是使用head_dim ** -0.5
            self.scale = qk_scale or head_dim ** -0.5

            # 初始化线性层q，用于转换查询向量（Query）
            self.q = nn.Linear(dim, dim, bias=qkv_bias)

            # 初始化dropout层，为注意力机制添加一层正则化
            self.attn_drop = nn.Dropout(attn_drop)

            # 初始化线性层kv，用于同时转换键向量（Key）和值向量（Value） 不使用偏置
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

            # 初始化一个3x3的分组卷积层，分组数与输入维度相同，可能是用于捕获局部空间信息
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        # 对模块中的所有子模块应用权重初始化函数，_init_weights是一个需要在类内部定义的方法
        self.apply(self._init_weights)

    # 定义初始化权重函数
    def _init_weights(self, m):
        # 定义一个初始化权重的函数，它接收一个模块m作为参数
        # 这个函数通常用于在神经网络模型初始化时，对不同的层设置不同的权重初始化策略

        if isinstance(m, nn.Linear):
            # 检查m是否是一个线性层

            trunc_normal_(m.weight, std=.02)
            # 使用截断正态分布初始化线性层的权重，标准差为0.02
            # 截断正态分布可以避免权重值过大或过小，有助于模型训练的稳定性

            if isinstance(m, nn.Linear) and m.bias is not None:
                # 再次检查m是否为线性层，并且该层有偏置项
                # 这里其实是一个冗余检查，因为已经在外部if条件中检查过m是线性层

                nn.init.constant_(m.bias, 0)
                # 将线性层的偏置项初始化为0

        elif isinstance(m, nn.LayerNorm):
            # 检查m是否是一个层归一化层

            nn.init.constant_(m.bias, 0)
            # 将层归一化层的偏置项初始化为0

            nn.init.constant_(m.weight, 1.0)
            # 将层归一化层的权重（通常是缩放因子）初始化为1.0
            # 这通常是层归一化的标准做法

        elif isinstance(m, nn.Conv2d):
            # 检查m是否是一个二维卷积层

            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 计算fan_out，即卷积核的元素总数
            # 它等于卷积核的尺寸（高x宽）乘以输出通道数

            fan_out //= m.groups
            # 如果卷积层使用了分组卷积，那么需要将fan_out除以分组数

            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 使用正态分布初始化卷积层的权重，均值为0，标准差为sqrt(2/fan_out)
            # 这是一个常用的初始化策略，称为He初始化，有助于保持网络各层的激活值和梯度在合理的范围内

            if m.bias is not None:
                # 检查卷积层是否有偏置项

                m.bias.data.zero_()
                # 如果存在偏置项，将其初始化为0

    # 定义前向传播函数，接收输入x，以及高H和宽W
    def forward(self, x, H, W):

        # 获取输入x的批次大小B、序列长度N和通道数C
        B, N, C = x.shape

        # 如果启用了通道注意力（ca_attention为1）
        if self.ca_attention == 1:

            # 将输入x通过线性层v进行转换，得到v
            v = self.v(x)

            # 将输入x通过线性层s进行转换，然后重塑并转置张量，得到s
            s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)

            # 遍历每个通道注意力头
            for i in range(self.ca_num_heads):

                # 动态获取对应的局部卷积层
                local_conv = getattr(self, f"local_conv_{i + 1}")

                # 获取当前通道注意力头的s
                s_i = s[i]

                # 对s_i进行卷积操作，然后重塑张量
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)

                # 如果是第一个通道注意力头
                if i == 0:

                    # 将s_i赋值给s_out
                    s_out = s_i

                # 如果不是第一个通道注意力头
                else:

                    # 将s_i沿着第2维拼接到s_out上
                    s_out = torch.cat([s_out, s_i], 2)

            # 将s_out重塑为4维张量
            s_out = s_out.reshape(B, C, H, W)

            # 对s_out进行一系列操作：先通过proj0进行组内卷积操作，然后经过bn进行批归一化，
            # 再通过act进行激活，最后通过proj1进行组间卷积操作
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))

            # 将s_out赋值给类属性modulator
            self.modulator = s_out

            # 将s_out重塑并转置张量
            s_out = s_out.reshape(B, C, N).permute(0, 2, 1)

            # 将s_out与v进行逐元素相乘，得到新的x
            x = s_out * v

        # 如果没有启用通道注意力
        else:

            # 将输入x通过线性层q进行转换，然后重塑并转置张量，得到查询向量q
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)

            # 将输入x通过线性层kv进行转换，然后重塑并转置张量，得到键向量k和值向量v
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)

            # 解构得到k和v
            k, v = kv[0], kv[1]

            # 计算注意力分数，通过矩阵乘法并乘以缩放因子
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # 对注意力分数应用softmax函数，得到注意力权重
            attn = attn.softmax(dim=-1)

            # 对注意力权重应用dropout层
            attn = self.attn_drop(attn)

            # 根据注意力权重计算加权和得到新的值向量，并加上局部卷积处理后的v
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
                self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                          N).transpose(
                    1, 2)

        # 将x通过线性层proj进行转换，可能是为了调整特征维度或进行进一步的特征融合
        x = self.proj(x)

        # 对x应用dropout层proj_drop，防止过拟合
        x = self.proj_drop(x)

        # 返回经过前向传播处理后的x
        return x


'''
第一阶段SAM模块的第一个子模块MHMC（多头混合卷积模块） 传入参数dim=64, ca_num_heads=4, sa_num_heads=-1, mlp_ratio=4., qkv_bias=True, qk_scale=None,use_layerscale=False, layerscale_value=1e-4, 
drop=0., attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm.eps=1e-6), ca_attention=1, expand_ratio=2，传入参数后，首先调用归一化层对输入进行层归一化，然后调用
Attention类，传入dim=64,ca_num_heads=ca_num_heads=4, sa_num_heads=sa_num_heads=-1, qkv_bias=qkv_bias=True, qk_scale=qk_scale=Nonne, attn_drop=attn_drop=0, proj_drop=drop=0, 
ca_attention=ca_attention=1, expand_ratio=expand_ratio=2

'''


class Block(nn.Module):
    # 定义Block类，继承自PyTorch的nn.Module

    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1, expand_ratio=2):
        # 初始化函数，接受多个参数来配置Block的属性
        # dim: 输入特征的维度
        # ca_num_heads: 通道注意力的头数
        # sa_num_heads: 自注意力的头数
        # mlp_ratio: 多层感知机(MLP)的隐藏层维度与输入维度的比例
        # qkv_bias: 查询(q), 键(k), 值(v)的偏置项是否启用
        # qk_scale: 查询和键的缩放因子
        # use_layerscale: 是否使用层缩放
        # layerscale_value: 层缩放的初始值
        # drop: 丢弃率，用于MLP和注意力之后的dropout操作
        # attn_drop: 注意力操作后的丢弃率
        # drop_path: 随机深度中的丢弃路径概率
        # act_layer: 激活函数层
        # norm_layer: 归一化层
        # ca_attention: 是否启用通道注意力，1表示启用，0表示不启用
        # expand_ratio: MLP的扩展比例

        super().__init__()
        # 调用父类nn.Module的初始化方法

        self.norm1 = norm_layer(dim)
        # 实例化一个归一化层，输入维度为dim

        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention,
            expand_ratio=expand_ratio)
        # 实例化一个Attention层，传入多个参数进行配置

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 注释：为随机深度设置丢弃路径，我们将会看到这是否比dropout更好

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 如果drop_path大于0，则实例化一个DropPath层,让输入随机置零，否则使用恒等映射

        self.norm2 = norm_layer(dim)
        # 再次实例化一个归一化层，输入维度为dim，这里采用的是partial(nn.LayerNorm.eps=1e-6) 层归一化则针对每个样本的特征进行求均值和标准差std，为了防止除以0，所以传入eps，让分母变为std+eps
        # norm_layer是一个新的函数，它等同于调用nn.LayerNorm并总是将eps参数设置为1e - 6,这样，每当您使用norm_layer来创建层归一化层时，您都不需要再单独指定eps参数，它会自动被设置为1e - 6

        mlp_hidden_dim = int(dim * mlp_ratio)
        # 计算MLP隐藏层的维度

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # 实例化一个Mlp层，传入输入特征维度、隐藏层维度、激活函数层nn.GELU和dropout率为0

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        # 初始化两个可学习参数gamma_1和gamma_2，用于层缩放

        if use_layerscale:  # 这里层缩放，但是这个smt模型不用.
            # 如果启用层缩放

            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            # 将gamma_1初始化为一个可学习的参数，其值基于layerscale_value和维度dim

            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            # 将gamma_2也初始化为一个可学习的参数，与gamma_1类似

        self.apply(self._init_weights)
        # 对Block中的所有子模块应用权重初始化函数_init_weights（该函数应在Block类定义之外实现）

    def _init_weights(self, m):
        # 定义一个方法，用于初始化网络层的权重
        # 参数m代表需要初始化的网络层

        if isinstance(m, nn.Linear):
            # 如果m是一个线性层（全连接层）

            trunc_normal_(m.weight, std=.02)
            # 使用截断正态分布初始化权重，标准差为0.02
            # trunc_normal_可能是一个自定义函数，用于截断正态分布初始化，确保权重值不会过大或过小

            if isinstance(m, nn.Linear) and m.bias is not None:
                # 如果m是线性层且该层有偏置项

                nn.init.constant_(m.bias, 0)
                # 将偏置项初始化为0

        elif isinstance(m, nn.LayerNorm):
            # 如果m是一个层归一化层

            nn.init.constant_(m.bias, 0)
            # 将层归一化层的偏置项初始化为0

            nn.init.constant_(m.weight, 1.0)
            # 将层归一化层的权重初始化为1.0
            # 在层归一化中，权重通常对应缩放因子gamma

        elif isinstance(m, nn.Conv2d):
            # 如果m是一个二维卷积层

            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 计算fan_out，即输出特征图的元素总数
            # 这通常用于确定权重的初始化标准差，以考虑网络层的宽度

            fan_out //= m.groups
            # 如果卷积层使用了分组卷积，则fan_out除以分组数

            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 使用正态分布初始化卷积层的权重，均值为0，标准差为sqrt(2/fan_out)
            # 这种初始化方法通常称为He初始化，有助于在训练初期保持网络层的激活值分布合理

            if m.bias is not None:
                # 如果卷积层有偏置项

                m.bias.data.zero_()
                # 将偏置项初始化为0

    def forward(self, x, H, W):
        # 定义forward方法，用于前向传播
        # x是输入数据，H和W可能是与输入数据相关的空间维度（如高度和宽度）

        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
        # 这行代码执行了以下操作：
        # 1. 使用self.norm1对输入x进行归一化处理，是层归一化（Layer Normalization） 这里就是针对每一个stage阶段的Patch_embed的输出进行正则化
        # 2. 将归一化后的x传递给self.attn（注意力模块），同时传入H和W作为额外的参数，
        # 3. 将注意力模块的输出乘以可学习的缩放因子self.gamma_1
        # 4. 使用self.drop_path进行随机路径丢弃（drop path），这是深度模型训练中的一种正则化技术，用于防止模型过拟合，只有第一个阶段的SAM模块中的MHMC不进行drop，因为drop[0]=0,只进行恒等映射
        # 5. 将上述结果加到原始输入x上，实现残差连接（residual connection），有助于缓解梯度消失问题

        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        # 这行代码执行了与上一行类似的操作，但这次是使用了MLP（多层感知机）模块
        # 1. 使用self.norm2对当前的x进行归一化处理
        # 2. 将归一化后的x传递给self.mlp（多层感知机模块），同时传入H和W作为额外的参数
        # 3. 将MLP的输出乘以另一个可学习的缩放因子self.gamma_2
        # 4. 同样地，使用self.drop_path进行随机路径丢弃
        # 5. 将结果加到上一步得到的x上，再次实现残差连接 每一个block对应一个drop_path,只有stage1的第一个block的drop_path=0，其他block的正常使用

        return x
        # 返回经过两次残差连接后的x，作为该层的输出


'''

调用OverlapPatchEmbed,则表明进入了第二阶段，传入输入[4,56*56,64],patch_size=3, stride=2, in_chans=64,embed_dim=128,然后通过二维卷积层nn.Conv2d(64, 128, kernel_size=patch_size=3, 
stride=2,padding=(patch_size[0] // 2, patch_size[1] // 2)=(1,1))   输出结果为：[4,128,(56-3+2*1+2)/2,(56-3+2*1+2)/2]=[4,128,28.5,28.5],向下取整为[4,128,28,28]
然后获取第一阶段后的高H=28,宽W=28, 将嵌入张量展平并转置，使其形状变为[4, 28*28, 128]，最后层归一化形状不变仍为[4, 28*28, 128]  最终返回x,H,W  将结果传入stage2的

再调用OverlapPatchEmbed,则表明进入了第三阶段，传入图片大小进行了修正img_size=28,atch_size=3, stride=2,in_chans=128,embed_dim=256,然后通过二维卷积层nn.Conv2d(128, 256, kernel_size=patch_size=3, 
stride=2,padding=(patch_size[0] // 2, patch_size[1] // 2)=(1,1))   输出结果为：[4,256,(28-3+2*1+2)/2,(28-3+2*1+2)/2]=[128,256,14.5,14.5],向下取整为[4,256,14,14]
然后获取第一阶段后的高H=14,宽W=14 将嵌入张量展平并转置，使其形状变为[4, 14*14, 256，最后层归一化形状不变仍为[4, 14*14, 256]  最终返回x,H,W

最后调用OverlapPatchEmbed,则表明进入了第四阶段，传入图片大小进行了修正img_size=14,patch_size=3,stride=2,in_chans=256,embed_dim=512,然后通过二维卷积层nn.Conv2d(256,512,kernel_size=patch_size=3,
stride=2,padding=(patch_size[0] // 2, patch_size[1] // 2)=(1,1))   输出结果为：[4,512,(14-3+2*1+2)/2,(14-3+2*1+2)/2]=[4,512,7.5,7.5],向下取整为[4,512,7,7]
然后获取第一阶段后的高H=7,宽W=7, 将嵌入张量展平并转置，使其形状变为[4, 7*7, 512]，最后层归一化形状不变仍为[4, 7*7, 512]  最终返回x,H,W

输入--->卷积映射--->展平、转置--->层归一化--->输出

'''


class OverlapPatchEmbed(nn.Module):
    """
    Imgs to Patch Embedding
    将图像转换为块嵌入
    """

    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        # 初始化函数，定义模型参数
        # img_size: 输入图像的大小，默认为224x224
        # patch_size: 块的大小，默认为3x3
        # stride: 块之间的步长，默认为2
        # in_chans: 输入图像的通道数，默认为3（RGB图像）
        # embed_dim: 嵌入空间的维度，默认为768,第一阶段64，第二阶段128，第三阶段256，第四阶段512

        super().__init__()
        # 调用父类nn.Module的初始化方法

        patch_size = to_2tuple(patch_size)
        # 将patch_size转换为元组，确保它是二维的

        img_size = to_2tuple(img_size)
        # 将img_size转换为元组，确保它是二维的

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # 定义一个二维卷积层，用于将图像分割成小块并嵌入到高维空间中
        # 这里卷积核的大小是patch_size，步长是stride，填充是patch_size的一半，以确保块之间有重叠

        self.norm = nn.LayerNorm(embed_dim)
        # 定义一个层归一化层，用于对嵌入进行归一化

        self.apply(self._init_weights)
        # 应用自定义的权重初始化方法到模型的所有子模块

    def _init_weights(self, m):
        # 自定义的权重初始化方法
        if isinstance(m, nn.Linear):
            # 如果子模块是线性层
            trunc_normal_(m.weight, std=.02)
            # 使用截断正态分布初始化权重，标准差为0.02
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                # 如果线性层有偏置项，则将其初始化为0

        elif isinstance(m, nn.LayerNorm):
            # 如果子模块是层归一化层
            nn.init.constant_(m.bias, 0)
            # 将层归一化层的偏置项初始化为0
            nn.init.constant_(m.weight, 1.0)
            # 将层归一化层的权重初始化为1.0

        elif isinstance(m, nn.Conv2d):
            # 如果子模块是二维卷积层
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 计算fan_out，即输出特征图的元素总数
            fan_out //= m.groups
            # 如果卷积层使用了分组卷积，则fan_out除以分组数
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 使用正态分布初始化卷积层的权重，均值为0，标准差为sqrt(2/fan_out)
            if m.bias is not None:
                m.bias.data.zero_()
                # 如果卷积层有偏置项，则将其初始化为0

    def forward(self, x):
        # 前向传播函数
        x = self.proj(x)
        # 通过投影卷积层将输入图像x转换为嵌入 stage2接收来自stage1的输出[4,56*56,64] 传入卷积层计算

        _, _, H, W = x.shape
        # 获取嵌入张量的形状，这里假设其形状为[B, C, H, W]   获取宽高为56

        x = x.flatten(2).transpose(1, 2)
        # 将嵌入张量展平并转置，使其形状变为[B, H*W, C]

        x = self.norm(x)
        # 对嵌入进行层归一化

        return x, H, W
        # 返回嵌入张量x，以及变换后的高度


'''
这里其实就是输入图像第一次进入网络时，首先通过的第一个stage的第一个模块stem  这里测试的时候输入为[4,3,224,224],然后定义SMT类时传入head_conv=3，dim=64,此时第一个卷积层Conv2d(3,64,kernal=3,stride=2,
padding=1,bias=False),然后通过第一个卷积层后输出为[4,64,(224-3+2*1+2)/2,(224-3+2*1+2)/2]=[4,64,112,112]。然后通过批量规范化层和激活函数ReLu层之后不会改变输入形状，依然为[4,64,112,112]。然后在
通过第二个卷积层Conv2d(64,64,kernal=2,stride=2)，输出为[4,64,(112-2+2*0+2)/2,(112-2+2*0+2)/2]=[4,64,56,56]，然后获取卷积层输出结果的高H和宽W,将输出x的最后两个维度展成一维的，所以输出[4,64,56*56]
再通过转置，使其形状变为[4, 56*56, 64]，最后通过LayerNorm对展平后的张量x进行层归一化，最终stem模块返回的结果是x(形状为[4, 56*56, 64])，H=56,W=56  进行了下采样输出[B,H/4*W/4,C1]

所以stem的传输过程如下：
    输入--->卷积层--->批量归一化--->激活函数ReLu--->卷积层--->展平、转置--->层归一化--->输出
'''


class Head(nn.Module):
    # 定义Head类，继承自PyTorch的nn.Module基类

    def __init__(self, head_conv, dim):
        # 初始化函数，接收两个参数：head_conv（卷积核大小）和dim（目标维度）
        super(Head, self).__init__()
        # 调用父类nn.Module的初始化方法

        stem = [
            nn.Conv2d(3, dim, head_conv, 2, padding=3 if head_conv == 7 else 1, bias=False),
            # 定义一个二维卷积层，输入通道数为3（RGB图像），输出通道数为dim
            # 卷积核大小为head_conv，步长为2，根据head_conv的大小决定填充为3或1
            # 不使用偏置项
            nn.BatchNorm2d(dim),  # 定义一个批归一化层，用于对卷积层的输出进行归一化
            nn.ReLU(True)  # 定义一个ReLU激活函数
        ]
        # 将上述层添加到stem列表中，形成stem网络的前半部分

        stem.append(nn.Conv2d(dim, dim, kernel_size=2, stride=2))
        # 向stem列表中添加另一个二维卷积层，输入和输出通道数均为dim
        # 卷积核大小为2，步长为2，用于进一步下采样特征图

        self.conv = nn.Sequential(*stem)
        # 将stem列表中的层按顺序组合成一个Sequential模型，并赋值给self.conv

        self.norm = nn.LayerNorm(dim)
        # 定义一个层归一化层，用于对嵌入进行归一化

        self.apply(self._init_weights)
        # 应用自定义的权重初始化方法到模型的所有子模块

    def _init_weights(self, m):
        # 自定义的权重初始化方法
        if isinstance(m, nn.Linear):
            # 如果子模块是线性层
            trunc_normal_(m.weight, std=.02)
            # 使用截断正态分布初始化权重，标准差为0.02
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                # 如果线性层有偏置项，则将其初始化为0

        elif isinstance(m, nn.LayerNorm):
            # 如果子模块是层归一化层
            nn.init.constant_(m.bias, 0)
            # 将层归一化层的偏置项初始化为0
            nn.init.constant_(m.weight, 1.0)
            # 将层归一化层的权重初始化为1.0

        elif isinstance(m, nn.Conv2d):
            # 如果子模块是二维卷积层
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 计算fan_out，即输出特征图的元素总数
            fan_out //= m.groups
            # 如果卷积层使用了分组卷积，则fan_out除以分组数
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 使用正态分布初始化卷积层的权重，均值为0，标准差为sqrt(2/fan_out)
            if m.bias is not None:
                m.bias.data.zero_()
                # 如果卷积层有偏置项，则将其初始化为0

    def forward(self, x):
        # 前向传播函数,传入输入x，其形状为[B,C,H,W]
        x = self.conv(x)
        # 将输入x通过self.conv中的卷积层和批归一化层进行处理

        _, _, H, W = x.shape
        # 获取处理后的张量x的形状，这里假设其形状为[B, C, H/4, W/4]

        x = x.flatten(2).transpose(1, 2)
        # 将张量x展平并转置，使其形状变为[B, H*W, C]

        x = self.norm(x)
        # 对展平后的张量x进行层归一化

        return x, H, W


'''
调用SMT类，传入参数embed_dims=[64, 128, 256, 512],  # 嵌入维度的列表，用于模型中的不同层 ca_num_heads=[4, 4, 4, -1],  # CA（通道种注意力机制）的头数列表 
sa_num_heads=[-1, -1, 8, 16],  # SA（空间注意力机制）的头数列表 mlp_ratios=[4, 4, 4, 2],  # 多层感知机（MLP）的扩展比例列表
qkv_bias=True,  # 查询（q）、键（k）和值（v）的偏置项是否启用 depths=[2, 2, 8, 1],  # 各层的深度 ca_attentions=[1, 1, 1, 0],  # 通道注意力机制是否启用的列表
head_conv=3,  # 头部卷积层的参数，可能控制某种卷积操作 expand_ratio=2,  # 可能的扩展比例，用于控制模型某些部分的宽度 **kwargs  # 其他关键字参数，它们可能用于SMT类的其他初始化参数
其他参数未改使用默认值。
'''


class SMT(nn.Module):

    # 初始化函数，当创建SMT类的实例时，这个函数会被调用,它接受多个参数，包括图像大小、输入通道数、类别数、嵌入维度、注意力头数、MLP比率、偏置项、缩放因子、层标准化值、丢弃率等。
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[8, 6, 4, 2],
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 8, 1], ca_attentions=[1, 1, 1, 0], num_stages=4, head_conv=3, expand_ratio=2, **kwargs):

        # 调用父类nn.Module的初始化函数
        super().__init__()

        # 保存类别数量
        self.num_classes = num_classes

        # 保存每个阶段的深度（层数）
        self.depths = depths

        # 保存总的阶段数
        self.num_stages = num_stages

        # 使用线性空间插值来生成每个阶段的drop path rate，这是一种随机深度衰减规则
        # torch.linspace生成从0到drop_path_rate的等差数列，长度为sum(depths)，然后转换为列表
        '''产生的列表是dpr=[0.0,
                          0.01666666753590107,
                          0.03333333507180214,
                          0.05000000447034836,
                          0.06666667014360428,
                          0.0833333358168602,
                          0.09999999403953552,
                          0.11666666716337204,
                          0.13333332538604736,
                          0.15000000596046448,
                          0.1666666716337204,
                          0.18333333730697632,
                          0.20000000298023224]'''
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # 初始化cur变量为0，用于记录当前处理的层数位置
        cur = 0

        # 遍历每个阶段
        for i in range(num_stages):
            # 如果当前是第一阶段
            if i == 0:
                # 初始化Head模块，用于将图像转化为嵌入向量
                patch_embed = Head(head_conv, embed_dims[i])  # 假设Head是一个自定义的模块，用于图像到嵌入的转换,返回值是x,其形状是[B,H/4*W/4,C1=64]
            else:
                # 初始化OverlapPatchEmbed模块，用于后续阶段的图像嵌入
                patch_embed = OverlapPatchEmbed(
                    img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),  # 如果是第一阶段，则使用原始图像大小；否则使用下采样后的尺寸
                    patch_size=3,  # patch大小
                    stride=2,  # 步长
                    in_chans=embed_dims[i - 1],  # 输入通道数，即上一阶段的嵌入维度
                    embed_dim=embed_dims[i]  # 输出嵌入维度
                )

            # 初始化Block模块的列表，每个阶段由多个Block组成
            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],  # 当前阶段的嵌入维度
                    ca_num_heads=ca_num_heads[i],  # 当前阶段的CA注意力头数
                    sa_num_heads=sa_num_heads[i],  # 当前阶段的SA注意力头数
                    mlp_ratio=mlp_ratios[i],  # 当前阶段MLP的缩放比例
                    qkv_bias=qkv_bias,  # qkv偏置项
                    qk_scale=qk_scale,  # qk缩放因子
                    use_layerscale=use_layerscale,  # 是否使用层标准化
                    layerscale_value=layerscale_value,  # 层标准化的值
                    drop=drop_rate,  # 丢弃率
                    attn_drop=attn_drop_rate,  # 注意力丢弃率
                    drop_path=dpr[cur + j],  # 路径丢弃率，用于随机深度
                    norm_layer=norm_layer,  # 标准化层
                    ca_attention=0 if i == 2 and j % 2 != 0 else ca_attentions[i],
                    # 根据条件设置CA注意力，这里是在第三个stage，并且是MSA模块中使用列表,其他阶段不使用通道注意力
                    expand_ratio=expand_ratio  # 扩张比率
                )
                # 为每个阶段构建depths[i]个Block
                for j in range(depths[i])
            ])

            # 初始化归一化层
            norm = norm_layer(embed_dims[i])

            # 更新cur的值，以便下一个阶段使用正确的drop_path
            cur += depths[i]

            # 将patch_embed、block和norm设置为类的属性，以便后续使用
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # 初始化分类头
        # 如果num_classes大于0，则使用线性层作为分类头；否则使用恒等映射（即不改变输入）
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # 对模型的所有层应用权重初始化函数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 定义一个方法用于初始化权重，该方法接受一个模块m作为参数

        if isinstance(m, nn.Linear):
            # 如果模块m是线性层（nn.Linear）
            trunc_normal_(m.weight, std=.02)
            # 使用截断正态分布初始化线性层的权重，标准差为0.02

            if isinstance(m, nn.Linear) and m.bias is not None:
                # 如果模块m是线性层且存在偏置项
                nn.init.constant_(m.bias, 0)
                # 将偏置项初始化为0

        elif isinstance(m, nn.LayerNorm):
            # 如果模块m是层归一化层（nn.LayerNorm）
            nn.init.constant_(m.bias, 0)
            # 将层归一化层的偏置项初始化为0
            nn.init.constant_(m.weight, 1.0)
            # 将层归一化层的权重初始化为1.0

        elif isinstance(m, nn.Conv2d):
            # 如果模块m是二维卷积层（nn.Conv2d）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 计算fan_out，即输出的特征数量，由卷积核大小与输出通道数决定

            fan_out //= m.groups
            # 如果卷积层使用了分组卷积，则根据分组数调整fan_out

            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 使用正态分布初始化卷积层的权重，标准差由fan_out决定，这是一种常用的He初始化方法

            if m.bias is not None:
                # 如果卷积层存在偏置项
                m.bias.data.zero_()
                # 将偏置项初始化为0

    def freeze_patch_emb(self):
        # 定义一个方法freeze_patch_emb，用于冻结模型中的patch_embed1层(也就是Head层)的参数，使其在训练过程中不再更新

        self.patch_embed1.requires_grad = False
        # 将patch_embed1层的requires_grad属性设置为False，意味着在反向传播时不会计算该层的梯度，从而冻结其参数

    # 使用@torch.jit.ignore装饰器，表明该方法在TorchScript编译过程中将被忽略
    # TorchScript是PyTorch的一个子集，用于优化模型以便部署到没有Python环境的设备上
    @torch.jit.ignore
    def no_weight_decay(self):
        # 定义一个方法no_weight_decay，返回一个集合，其中包含不应应用权重衰减的层名
        # 权重衰减是一种正则化技术，用于防止模型过拟合
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}
        # 返回一个包含位置嵌入（position embeddings）和分类标记（classification token）的集合
        # 这些层的权重通常不应使用权重衰减，因为它们通常不是通过训练来学习的，而是固定或按特定方式初始化的

    def get_classifier(self):
        # 定义一个方法get_classifier，返回模型的分类器层
        return self.head
        # 返回模型的head属性，这通常是模型的最后一层，用于分类任务

    def reset_classifier(self, num_classes, global_pool=''):
        # 定义一个方法reset_classifier，用于重置模型的分类器层以处理新的类别数量
        self.num_classes = num_classes
        # 设置模型的num_classes属性，用于记录新的类别数量

        # 根据新的类别数量num_classes来重新初始化模型的head层
        # 如果num_classes大于0，则使用nn.Linear创建一个新的线性层作为分类器，
        # 输入维度是模型的embed_dim，输出维度是新的类别数量num_classes
        # 如果num_classes不大于0（即0或负数），则使用nn.Identity作为分类器，实际上不改变输入
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # 定义forward_features方法，该方法负责提取模型的特征，并返回不同层的特征输出

        B = x.shape[0]
        # 获取输入x的批次大小B

        layer_features = []
        # 初始化一个空列表，用于存储不同层的特征

        for i in range(self.num_stages):
            # 遍历模型的每一个阶段

            patch_embed = getattr(self, f"patch_embed{i + 1}")
            # 通过字符串拼接和getattr函数获取当前阶段的patch_embed层

            block = getattr(self, f"block{i + 1}")
            # 通过字符串拼接和getattr函数获取当前阶段的block层（可能包含多个基本块）

            norm = getattr(self, f"norm{i + 1}")
            # 通过字符串拼接和getattr函数获取当前阶段的归一化层

            x, H, W = patch_embed(x)
            # 调用patch_embed层处理输入x，返回处理后的x以及新的特征图高H和宽W

            for blk in block:
                # 遍历当前阶段的每一个基本块

                x = blk(x, H, W)
                # 调用基本块处理输入x，注意这里还传入了特征图的高H和宽W作为参数

            x = norm(x)
            # 调用归一化层处理基本块输出的特征x

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # 重新整理x的形状，并进行维度置换，使其适应后续操作或存储

            layer_features.append(x)
            # 将处理后的特征x添加到layer_features列表中

        return layer_features
        # 返回包含所有阶段特征输出的layer_features列表

    def forward(self, x):
        # 定义forward方法，这是模型在推理或训练时调用的主要方法

        x = self.forward_features(x)
        # 调用forward_features方法提取特征，并将结果赋值给x

        # x = self.head(x)
        # 原本这里可能是将提取的特征x传递给分类头self.head进行处理，但在这段代码中，这一行被注释掉了

        return x
        # 返回提取的特征x，因为head部分被注释了，所以返回的是特征的列表，而不是最终的分类结果


'''
深度可分离卷积（Depthwise Separable Convolution）是一种特殊的卷积操作，主要用于减少模型的计算量和参数量，从而在保持一定性能的同时，提高模型的效率,它主要应用在轻量级模型设计中.

深度可分离卷积可以看作是两个步骤的组合：深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）

深度卷积（Depthwise Convolution）：

这个步骤对每个输入通道独立地进行空间卷积。如果输入有M个通道，那么将使用M个不同的卷积核，每个卷积核负责一个通道。这样，每个输出通道都仅与一个输入通道相关联
这与传统的卷积操作不同，传统的卷积操作会使用一个卷积核来同时考虑多个输入通道的信息
深度卷积的主要优点是减少了计算量和参数量，因为它没有跨通道进行卷积操作
逐点卷积（Pointwise Convolution）：

这一步是一个标准的1x1卷积操作，它的作用是将深度卷积的输出在通道维度上进行组合，以生成新的特征表示
逐点卷积增加了模型的非线性，并允许不同通道之间的信息交互
尽管1x1卷积会增加一些计算量和参数量，但与传统的卷积操作相比，它仍然是非常高效的
通过组合深度卷积和逐点卷积，深度可分离卷积能够在保持一定性能的同时，显著减少模型的计算量和参数量,这使得深度可分离卷积在资源受限的环境（如移动设备或嵌入式设备）中特别有用，同时也为设计更轻量级的模型提供了有效的工具

总的来说，深度可分离卷积是一种高效的卷积操作，通过分解传统的卷积操作，它在减少计算量和参数量的同时，仍然能够保持模型的性能。这使得它在各种应用场景中，特别是资源受限的环境中，具有广泛的应用前景
'''


class DWConv(nn.Module):
    # 定义一个名为DWConv的类，继承自PyTorch的nn.Module基类  通过深度可分离卷积后的输出和输入是一样的形状，没有改变

    def __init__(self, dim=768):
        # 初始化函数，接受一个参数dim，表示输入和输出的通道数，默认为768
        super(DWConv, self).__init__()
        # 调用父类nn.Module的初始化方法

        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        # 定义一个二维卷积层，输入和输出通道数均为dim
        # 卷积核大小为3，步长为1，填充为1
        # 使用bias偏置项
        # groups参数设置为dim，实现深度可分离卷积，即每个输入通道使用单独的卷积核

    def forward(self, x, H, W):
        # 前向传播函数，接收输入张量x以及高H和宽W

        B, N, C = x.shape
        # 获取输入张量x的形状，这里假设其形状为[B, N, C]，其中B是批大小，N是特征图的元素数，C是通道数

        x = x.transpose(1, 2).view(B, C, H, W)
        # 将x的维度1和维度2转置，并重新调整形状为[B, C, H, W]，以便进行二维卷积操作

        x = self.dwconv(x)
        # 将调整形状后的x通过深度可分离卷积层进行处理

        x = x.flatten(2).transpose(1, 2)
        # 将卷积后的x展平并转置，使其形状再次变为[B, N, C]

        return x
        # 返回处理后的张量x


'''
这段代码定义了一个名为build_transforms的函数，该函数用于构建一系列图像预处理变换，这些变换通常用于深度学习模型的训练或测试阶段。下面是对这段代码的逐行注释：
'''


def build_transforms(img_size, center_crop=False):
    # 定义一个函数build_transforms，它接受两个参数：img_size（目标图像大小）和center_crop（是否进行中心裁剪，默认为False）

    t = []
    # 初始化一个空列表t，用于存放变换步骤

    if center_crop:
        # 如果center_crop为True，则执行以下中心裁剪的步骤

        size = int((256 / 224) * img_size)
        # 计算中心裁剪前的图像大小。这里假设原始图像大小与ImageNet上的图像大小（224x224）成比例，
        # 然后根据这个比例计算出一个更大的尺寸（256/224倍于目标大小），用于先调整图像大小再进行中心裁剪

        t.append(
            transforms.Resize(size, interpolation=str_to_pil_interp('bicubic'))
        )
        # 将图像大小调整为计算出的size，使用双三次插值（bicubic）作为插值方法
        # 注意：这里使用了str_to_pil_interp函数来将字符串'bicubic'转换为PIL库所需的插值模式，但代码中并未给出这个函数的定义

        t.append(
            transforms.CenterCrop(img_size)
        )
        # 在调整大小后的图像上进行中心裁剪，裁剪出img_size大小的图像区域

    else:
        # 如果center_crop为False，则直接调整图像大小，不进行中心裁剪

        t.append(
            transforms.Resize(img_size, interpolation=str_to_pil_interp('bicubic'))
        )
        # 直接将图像大小调整为img_size，使用双三次插值（bicubic）作为插值方法

    t.append(transforms.ToTensor())
    # 将PIL图像或NumPy ndarray转换为PyTorch张量，并自动将像素值缩放到[0.0, 1.0]范围

    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    # 对图像进行标准化处理，使用ImageNet数据集的默认均值和标准差
    # 注意：这里使用了IMAGENET_DEFAULT_MEAN和IMAGENET_DEFAULT_STD，但代码中并未给出这两个变量的定义

    return transforms.Compose(t)
    # 使用transforms.Compose将上述所有变换步骤组合成一个单一的变换，并返回这个组合变换


'''
这句话描述了一个图像处理任务中的步骤，具体涉及到图像大小的调整和使用的插值方法,我们可以分步解析这句话：图像大小调整为img_size：
这意味着我们要改变图像的尺寸,img_size 是一个变量，它代表目标尺寸。例如，如果 img_size 是 (500, 500)，那么我们的目标是把图像调整为宽度500像素和高度500像素
调整图像大小通常涉及到缩放操作，也就是放大或缩小图像。在缩放过程中，由于原始图像的像素与目标尺寸的像素可能不完全对应，所以需要使用某种插值方法来估算新像素的值
使用双三次插值（bicubic）作为插值方法：
    插值是一种估计方法，用于估算未知或缺失数据,在图像处理中，插值用于估算在缩放、旋转等变换过程中产生的新像素的值
    双三次插值（bicubic interpolation）是一种插值方法，它使用邻近的16个像素点（4x4的像素网格）的灰度值来估算新像素的值
    与线性插值（只考虑邻近的两个点）或双线性插值（考虑邻近的4个点）相比，双三次插值通常能产生更平滑、更自然的图像，特别是在放大图像时
'''


def build_transforms4display(img_size, center_crop=False):
    # 定义一个函数build_transforms4display，用于构建图像显示的预处理变换
    # 它接受两个参数：img_size（目标图像大小）和center_crop（是否进行中心裁剪，默认为False）

    t = []
    # 初始化一个空列表t，用于存放变换步骤

    if center_crop:
        # 如果center_crop为True，执行以下中心裁剪的步骤

        size = int((256 / 224) * img_size)
        # 计算中心裁剪前的图像大小。与build_transforms函数类似，这里先按照比例放大图像大小

        t.append(
            transforms.Resize(size, interpolation=str_to_pil_interp('bicubic'))
        )
        # 将图像大小调整为计算出的size，使用双三次插值（bicubic）作为插值方法

        t.append(
            transforms.CenterCrop(img_size)
        )
        # 在调整大小后的图像上进行中心裁剪，裁剪出img_size大小的图像区域

    else:
        # 如果center_crop为False，则直接调整图像大小，不进行中心裁剪

        t.append(
            transforms.Resize(img_size, interpolation=str_to_pil_interp('bicubic'))
        )
        # 直接将图像大小调整为img_size，使用双三次插值（bicubic）作为插值方法

    t.append(transforms.ToTensor())
    # 将PIL图像或NumPy ndarray转换为PyTorch张量，并自动将像素值缩放到[0.0, 1.0]范围
    # 这一步是为了适应PyTorch模型的数据输入格式

    return transforms.Compose(t)
    # 使用transforms.Compose将上述所有变换步骤组合成一个单一的变换，并返回这个组合变换


def smt_t(pretrained=False, **kwargs):
    # 定义一个函数smt_t，它接受一个可选参数pretrained（默认为False）和任意数量的关键字参数kwargs
    # 如果pretrained为True，可能意味着模型会被初始化为预训练权重，但这里并没有直接实现这一逻辑

    model = SMT(
        # 创建一个SMT模型实例，并传入以下参数来配置模型的结构：

        embed_dims=[64, 128, 256, 512],  # 嵌入维度的列表，用于模型中的不同层

        ca_num_heads=[4, 4, 4, -1],  # CA（通道种注意力机制）的头数列表

        sa_num_heads=[-1, -1, 8, 16],  # SA（空间注意力机制）的头数列表

        mlp_ratios=[4, 4, 4, 2],  # 多层感知机（MLP）的扩展比例列表

        qkv_bias=True,  # 查询（q）、键（k）和值（v）的偏置项是否启用

        depths=[2, 2, 8, 1],  # 各层的深度

        ca_attentions=[1, 1, 1, 0],  # 通道注意力机制是否启用的列表

        head_conv=3,  # 头部卷积层的参数，可能控制某种卷积操作

        expand_ratio=2,  # 可能的扩展比例，用于控制模型某些部分的宽度

        **kwargs  # 其他关键字参数，它们可能用于SMT类的其他初始化参数
    )

    model.default_cfg = _cfg()
    # 为模型实例设置一个default_cfg属性，该属性包含ViT模型的默认配置或元数据

    return model
    # 返回创建的SMT模型实例


def smt_s(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[4, 4, 4, 2],
        qkv_bias=True, depths=[3, 4, 18, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


def smt_b(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2],
        qkv_bias=True, depths=[4, 6, 28, 2], ca_attentions=[1, 1, 1, 0], head_conv=7, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


def smt_l(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[96, 192, 384, 768], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2],
        qkv_bias=True, depths=[4, 6, 28, 4], ca_attentions=[1, 1, 1, 0], head_conv=7, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


if __name__ == '__main__':
    # 如果这个脚本是作为一个独立的程序运行，而不是被其他模块导入，则执行下面的代码块

    import torch

    # 导入PyTorch库，PyTorch是一个用于深度学习的开源库

    model = smt_t()
    # 创建一个名为smt_s的模型实例,注意：这里假定smt_s是一个先前定义的函数或类，用于创建模型

    model = model.cuda()
    # 将模型转移到GPU上。如果机器上没有CUDA支持的GPU，这行代码会抛出错误

    input = torch.rand(4, 3, 224, 224).cuda()
    # 创建一个形状为(4, 3, 224, 224)的随机张量，代表一个批次包含4个图像，每个图像有3个颜色通道，尺寸为224x224
    # 然后将这个张量也转移到GPU上

    output = model(input)
    # 将上面创建的随机输入张量传递给模型，得到模型的输出

    print(model)
    # 打印模型的结构和参数

    ### thop cal ###
    # 下面这部分代码被注释掉了，看起来是原本用于计算模型FLOPs和参数数量的，使用的是thop库
    # 如果需要计算，可以取消注释并确保已经安装了thop库

    # input_shape = (1, 3, 384, 384) # 输入的形状
    # input_data = torch.randn(*input_shape)
    # 创建一个新的随机输入张量，用于thop库计算模型复杂度

    # macs, params = profile(model, inputs=(input_data,))
    # 使用thop库的profile函数计算模型的乘法累加操作数（MACs，通常用于表示FLOPs）和参数数量

    # print(f"FLOPS: {macs / 1e9:.2f}G")
    # 打印以G（十亿）为单位的FLOPs

    # print(f"params: {params / 1e6:.2f}M")
    # 打印以M（百万）为单位的参数数量

    ### ptflops cal ###
    # 下面是使用ptflops库计算模型复杂度的代码

    flops_count, params_count = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                          print_per_layer_stat=False)
    # 调用get_model_complexity_info函数，计算模型的浮点运算次数（FLOPs）和参数数量
    # 输入参数包括模型、输入形状，以及是否以字符串形式返回结果和是否打印每层统计信息

    print('flops: ', flops_count)
    # 打印模型的浮点运算次数

    print('params: ', params_count)
    # 打印模型的参数数量
