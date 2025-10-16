import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable


# '''
# 这段代码是用于计算前景特征（embedded_fg）和背景特征（embedded_bg）之间的余弦相似度的函数
# '''
# 定义了一个名为 cos_simi 的函数，它接受两个参数：embedded_fg 和 embedded_bg，这两个参数应该是张量（tensor），分别代表前景和背景的嵌入特
def cos_simi(embedded_fg, embedded_bg):
    # 使用 PyTorch 的 F.normalize 函数将前景特征张量 embedded_fg 在维度 1（通常是特征维度）上进行规范化，使其每个元素的向量长度为1，即进行单位化处理
    embedded_fg = F.normalize(embedded_fg, dim=1)
    # 同样地，对背景特征张量 embedded_bg 进行单位化处理
    embedded_bg = F.normalize(embedded_bg, dim=1)
    # 使用 PyTorch 的 torch.matmul 函数计算前景特征和背景特征的矩阵乘法。embedded_bg.T 表示背景特征张量的转置，这样乘法的结果是一个前景特征和背景特征之间的相似度矩阵
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    # 最后，使用 torch.clamp 函数将相似度矩阵中的值限制在 [0.0005, 0.9995] 范围内。这样做可以防止数值计算中的极小或极大值对模型造成不良影响，同时确保相似度值在合理的范围内。函数返回这个处理后的相似度矩阵
    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C


# Minimize Similarity, e.g., push representation of foreground and background apart. 这一行是注释，
# 说明了这个类的目的：最小化相似度，例如，将前景和背景的表示推得更远
# 这段代码定义了一个名为 SimMinLoss 的 PyTorch 损失函数类，用于最小化前景特征和背景特征之间的相似度
class SimMinLoss(nn.Module):
    # __init__ 方法是类的构造函数，用于初始化 SimMinLoss 类的实例。
    # 它接受两个参数：metric 用于指定相似度的度量方式，默认为 'cos'（余弦相似度），reduction 用于指定损失的聚合方式，默认为 'mean'（求平均）
    # super(SimMinLoss, self).__init__() 调用父类的构造函数
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    # forward 方法定义了损失函数的前向传播逻辑。它接受两个参数：embedded_bg 和 embedded_fg，分别代表背景和前景的嵌入特征
    # 函数的注释说明了参数的形状，[N, C] 表示每个特征向量有 N 个样本和 C 个特征维度
    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        # 如果指定的度量方式是 'l2'（欧几里得距离），则抛出 NotImplementedError 异常，因为当前代码只实现了余弦相似度
        if self.metric == 'l2':
            raise NotImplementedError
        # 如果指定的度量方式是 'cos'，则调用之前定义的 cos_simi 函数来计算余弦相似度，然后计算损失值，这里使用了 -log(1 - sim)，
        # 这是一种常见的余弦相似度损失函数，目的是使相似度尽可能小
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        # 如果指定的度量方式既不是 'l2' 也不是 'cos'，则同样抛出 NotImplementedError 异常
        else:
            raise NotImplementedError

        # 根据 reduction 参数的值，对损失进行聚合。如果 reduction 是 'mean'，则返回损失的平均值；如果是 'sum'，则返回损失的总和
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


# 这段代码定义了一个名为 SimMaxLoss 的 PyTorch 损失函数类，用于最大化背景特征之间的相似度
# Maximize Similarity, e.g., pull representation of background and background together.
# 这一行是注释，说明了这个类的目的：最大化相似度，例如，将背景的表示拉近
# 定义了一个名为 SimMaxLoss 的类，它继承自 PyTorch 的 nn.Module 类，是一个自定义的损失函数类
class SimMaxLoss(nn.Module):
    # __init__ 方法是类的构造函数，用于初始化 SimMaxLoss 类的实例
    # 它接受三个参数：metric 用于指定相似度的度量方式，默认为 'cos'（余弦相似度），alpha 是一个用于调整排名权重的超参数，默认为 0.25，
    # reduction 用于指定损失的聚合方式，默认为 'mean'（求平均）
    # super(SimMaxLoss, self).__init__() 调用父类的构造函数
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    # forward 方法定义了损失函数的前向传播逻辑。它接受一个参数：embedded_bg，代表背景的嵌入特征
    # 函数的注释说明了参数的形状，[N, C] 表示每个特征向量有 N 个样本和 C 个特征维度
    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        # 如果指定的度量方式是 'l2'（欧几里得距离），则抛出 NotImplementedError 异常，因为当前代码只实现了余弦相似度
        if self.metric == 'l2':
            raise NotImplementedError

        # 如果指定的度量方式是 'cos'，则调用 cos_simi 函数来计算背景特征自身的余弦相似度
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            # 计算损失值，这里使用了 -log(sim)，然后确保损失值不为负，因为对数函数的定义域是正数
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            # 对每个样本的相似度进行排序，并计算每个相似度值的排名，排名从 1 开始
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            # 计算排名权重，使用指数函数和 alpha 参数来调整权重，然后将权重应用于损失值
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        # 如果指定的度量方式既不是 'l2' 也不是 'cos'，则同样抛出 NotImplementedError 异常
        else:
            raise NotImplementedError

        # 根据 reduction 参数的值，对损失进行聚合
        # 如果 reduction 是 'mean'，则返回损失的平均值；如果是 'sum'，则返回损失的总和
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss_v1(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss_v1, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            sim_exp = torch.exp(sim)
            weights = sim_exp / torch.sum(sim_exp, dim=1, keepdim=True)
            loss = loss * weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


def iou_loss(pred, mask):
    """
    计算预测值与掩码之间的IoU损失

    参数:
        pred (torch.Tensor): 预测的输出张量，形状通常为 [batch_size, num_classes, height, width]
        mask (torch.Tensor): 真实掩码张量，形状通常为 [batch_size, 1, height, width]

    返回:
        torch.Tensor: 预测值与掩码之间的IoU损失的均值。
    """

    # 将预测值通过Sigmoid函数映射到0到1之间，以表示概率
    pred = torch.sigmoid(pred)

    # 计算预测值与掩码的重叠部分（交集）
    # 通过逐元素相乘然后沿着高度和宽度维度求和得到
    inter = (pred * mask).sum(dim=(2, 3))

    # 计算预测值与掩码的总和（并集）
    # 预测值和掩码相加后沿着高度和宽度维度求和，得到并集区域的大小
    union = (pred + mask).sum(dim=(2, 3))

    # 计算IoU（Intersection over Union，交并比）
    # 注意这里的IoU计算公式稍有不同，常规公式是交集除以并集，这里通过减法进行了平滑处理以避免除数为0的情况
    # 实际上，这种计算方法与标准的IoU计算公式稍有差异，但用于损失函数时可能更稳定
    iou = 1 - (inter + 1) / (union - inter + 1)

    # 返回IoU损失的均值，通过求所有batch中IoU的平均值得到
    return iou.mean()


def cross_entropy2d_edge(input, target, reduction='mean'):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


# --------------------------------------------SSIM Loss-------------------------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel):
    # 使用卷积操作计算平均值
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # 计算方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    # 计算协方差矩阵
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean()


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel)


class IOU(nn.Module):
    def __init__(self, reduction='mean'):
        super(IOU, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1

            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)

        return IoU / b
