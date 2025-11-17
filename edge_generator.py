"""
@File: gen_edge.py
@Time: 2022/11/6
@Author: rp
@Software: PyCharm

"""

import cv2
import os

import numpy as np



# def Edge_Extract(root):
#     img_root = os.path.join(root, 'GT')
#     edge_root = os.path.join(root, 'Edge')
#
#     if not os.path.exists(edge_root):
#         os.mkdir(edge_root)
#
#     file_names = os.listdir(img_root)
#     img_name = []
#
#     for name in file_names:
#         print(f'Generate Edge Image {name} successful!')
#         if not name.endswith('.png'):
#             assert "This file %s is not PNG" % (name)
#         img_name.append(os.path.join(img_root, name[:-4] + '.png'))
#
#     index = 0
#     for image in img_name:
#         img = cv2.imread(image, 0)
#         cv2.imwrite(edge_root + '/' + file_names[index], cv2.Canny(img, 30, 100))
#         index += 1
#     return 0
#
#
# if __name__ == '__main__':
#     root = r'D:/Code/BGNet-master/data/TestDataset/NC4K/'
#     Edge_Extract(root)

import cv2
import os
import numpy as np
#
#
def Edge_Extract(root):
    img_root = os.path.join(root, 'GT')
    edge_root = os.path.join(root, 'Edge')

    if not os.path.exists(edge_root):
        os.makedirs(edge_root)  # 使用makedirs更安全

    file_names = os.listdir(img_root)

    for name in file_names:
        if not name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        print(f'Processing: {name}')

        # 读取图像
        img_path = os.path.join(img_root, name)
        img = cv2.imread(img_path, 0)  # 灰度模式读取

        if img is None:
            print(f"无法读取图像: {name}")
            continue

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(img, (3, 3), 0)

        # Canny边缘检测 - 调整阈值
        # 对于分割掩码，可以适当提高阈值
        low_threshold = 30
        high_threshold = 100
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        # 边缘膨胀（加粗边缘）- 增强膨胀效果
        kernel = np.ones((3, 3), np.uint8)  # 使用更大的核
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)  # 增加迭代次数

        # 可选：将边缘转换为纯白色(255)和纯黑色(0)
        # 确保边缘是纯白色
        _, binary_edges = cv2.threshold(dilated_edges, 127, 255, cv2.THRESH_BINARY)

        # 保存结果 - 使用无损压缩
        output_path = os.path.join(edge_root, name)
        # 添加参数确保保存质量
        cv2.imwrite(output_path, binary_edges, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print(f'Saved: {output_path}')

    return 0


# def Edge_Extract(root):
#     img_root = os.path.join(root, 'GT')
#     edge_root = os.path.join(root, 'Edge')
#
#     if not os.path.exists(edge_root):
#         os.mkdir(edge_root)
#
#     file_names = os.listdir(img_root)
#
#     for name in file_names:
#         print(f'Generate Edge Image {name} successful!')
#         if not name.endswith('.png'):
#             assert "This file %s is not PNG" % (name)
#
#         img_path = os.path.join(img_root, name)
#         img = cv2.imread(img_path, 0)
#
#         # 方法3：多步骤细化
#         # 1. 先进行高斯模糊减少噪声
#         blurred = cv2.GaussianBlur(img, (3, 3), 0)
#
#         # 2. 使用更高的Canny阈值
#         edges = cv2.Canny(blurred, 30, 100)
#
#         # 3. 形态学腐蚀细化线条
#         kernel = np.ones((1, 1), np.uint8)
#         thin_edges = cv2.erode(edges, kernel, iterations=1)
#
#         # 4. 可选：降低亮度（如果太亮）
#         # 将白色(255)变为灰色(128)
#         thin_edges = np.where(thin_edges == 255, 128, 0).astype(np.uint8)
#
#         cv2.imwrite(os.path.join(edge_root, name), thin_edges)
#
#     return 0


if __name__ == '__main__':
    root = r'D:/Code/data/TestDataset/CAMO/'
    Edge_Extract(root)

#
