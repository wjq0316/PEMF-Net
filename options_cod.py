import argparse  # 导入Python的命令行参数解析库argparse，用于从命令行接收参数

# 创建一个解析对象
parser = argparse.ArgumentParser()

# 添加命令行参数：epoch，类型为整数，默认值为300，用于指定训练的轮数（epoch）
parser.add_argument('--epoch', type=int, default=200, help='epoch number')

# 添加命令行参数：lr，类型为浮点数，默认值为5e-5，即0.00005，用于指定学习率
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')  # 5e-5

# 添加命令行参数：batchsize，类型为整数，默认值为8，用于指定训练时每个批次的大小
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')

# 添加命令行参数：trainsize，类型为整数，默认值为384，用于指定训练时数据集中图像的分辨率大小为384*384
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')

# 添加命令行参数：clip，类型为浮点数，默认值为0.5，用于指定梯度裁剪的阈值
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

# 添加命令行参数：decay_rate，类型为浮点数，默认值为0.1，用于指定学习率衰减的比率
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')

# 添加命令行参数：decay_epoch，类型为整数，默认值为100，用于指定每多少轮（epoch）衰减一次学习率
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')  # 100

# 添加命令行参数：load，类型为字符串，默认值为'./smt_tiny.pth'，用于指定从哪个检查点文件加载模型
parser.add_argument('--load', type=str, default='./smt_tiny.pth', help='train from checkpoints')

# 添加命令行参数：gpu_id，类型为字符串，默认值为'0'，用于指定训练时使用的GPU编号
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# 添加命令行参数：rgb_root，类型为字符串，默认值为'./COD_datasets/TrainDataset/Imgs/'，用于指定训练RGB图像的根目录
parser.add_argument('--rgb_root', type=str, default='./CodDataset/TrainDataset/Imgs/',
                    help='the training rgb images root')

# 添加命令行参数：gt_root，类型为字符串，默认值为'./COD_datasets/TrainDataset/GT/'，用于指定训练时真实标签图像的根目录
parser.add_argument('--gt_root', type=str, default='./CodDataset/TrainDataset/GT/',
                    help='the training gt images root')

# 添加命令行参数：edge_root，类型为字符串，默认值为'./COD_datasets/TrainDataset/GT/'，用于指定训练时真实标签图像的根目录
parser.add_argument('--edge_root', type=str, default='./CodDataset/TrainDataset/Edge/',
                    help='the training edge images root')

# 添加命令行参数：test_rgb_root，类型为字符串，默认值为'./COD_datasets/TestDataset/CAMO/Imgs/'，用于指定测试RGB图像的根目录
parser.add_argument('--test_rgb_root', type=str, default='./CodDataset/TestDataset/CAMO/Imgs/',
                    help='the test gt images root')

# 添加命令行参数：test_gt_root，类型为字符串，默认值为'./COD_datasets/TestDataset/CAMO/GT/'，用于指定测试时真实标签图像的根目录
parser.add_argument('--test_gt_root', type=str, default='./CodDataset/TestDataset/CAMO/GT/',
                    help='the test gt images root')

# 添加命令行参数：save_path，类型为字符串，默认值为'./cpts/'，用于指定保存模型和日志的路径
parser.add_argument('--save_path', type=str, default='./out/', help='the path to save models and logs')

# 解析命令行参数，并将结果存储在opt对象中
opt = parser.parse_args()
