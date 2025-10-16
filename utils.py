def clip_gradient(optimizer, grad_clip):
    """
    梯度裁剪函数，用于防止梯度爆炸问题
    将模型参数的梯度限制在[-grad_clip, grad_clip]范围内

    Args:
        optimizer (torch.optim.Optimizer): 优化器对象，包含模型参数和梯度
        grad_clip (float): 梯度裁剪的阈值

    Returns:
        None
    """
    # 遍历优化器中的参数组，也可以理解为是每一层
    for group in optimizer.param_groups:
        # 遍历参数组中的模型参数
        for param in group['params']:
            # 检查参数是否存在梯度
            if param.grad is not None:
                # 使用clamp_方法将梯度限制在[-grad_clip, grad_clip]范围内
                # clamp_方法会原地修改数据，将超出范围的值设置为边界值
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    """
    调整优化器中的学习率

    Args:
        optimizer (torch.optim.Optimizer): 优化器对象
        init_lr (float): 初始学习率
        epoch (int): 当前训练的轮数（epoch）
        decay_rate (float, optional): 学习率衰减率，默认为0.1
        decay_epoch (int, optional): 每隔多少个epoch进行学习率衰减，默认为30

    Returns:
        float: 调整后的学习率
    """

    # 计算学习率衰减系数，使用指数衰减
    decay = decay_rate ** (epoch // decay_epoch)

    # 遍历优化器中的每个参数组
    for param_group in optimizer.param_groups:
        # 将当前参数组的学习率设置为初始学习率乘以衰减系数
        param_group['lr'] = decay * init_lr

        # 获取调整后的学习率（虽然这一步在这里是多余的，因为我们已经设置了它）
        lr = param_group['lr']

    # 返回当前调整后的学习率（虽然这里只返回了最后一个参数组的学习率，通常所有参数组的学习率都是一样的）
    return lr



import torch
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')





