# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年09月21日
"""

import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def get_parser():
    """
    Get three arguments: "number", "keyword" and "scale", and they are all needed

    :return: parser
    """
    par = argparse.ArgumentParser()

    par.add_argument('-n', '--number', required=True, type=int,
                     help='Input the number of the picture you wanna process (jpg format needed).')
    par.add_argument('-k', '--keyword', choices=['otsu', 'adaptive'], required=True,
                     help=r'Choose the algorithm of the threshold function between "otsu" and "adaptive".')
    par.add_argument('-s', '--scale', required=True, nargs='+',
                     help='Input m n to divide the picture into m × n blocks.')
    par.add_argument('-o', '--output', required=True,
                     help='Input o to change the number of the output picture.')

    return par


def threshold(img, keyword='otsu', max_val=200, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
              threshold_type=cv2.THRESH_BINARY, block_size=5, c=3):
    """
    Choose Binarization Methods between "otsu" and "adaptive"
    :param img: Input grey image
    :param keyword: 'otsu' or 'adaptive'
    :param max_val:
    :param adaptive_method:
    :param threshold_type:
    :param block_size:
    :param c:
    :return: None
    """
    if keyword == 'adaptive':
        th = cv2.adaptiveThreshold(img, max_val, adaptive_method, threshold_type, block_size, c)
        return th
    elif keyword == 'otsu':
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    else:
        print('Keyword Error!')


# 将图片分为m×n块，返回一个5维BGR矩阵，和一个4维二值图矩阵
def divide(img, m, n):
    """
    Divide an image into m × n blocks and return a 5-dim BGR matrix and a 4-dim grey matrix,
    or just a 4-dim grey matrix
    :param img: Any
    :param m: m raws
    :param n: n columns
    :return: divided, divided_grey
    """
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / m)
    grid_w = int(w * 1.0 / n)

    # 计算满足整除关系的高与宽
    h = grid_h * m
    w = grid_w * n

    # 进行图像缩放
    resized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.linspace(0, w, n + 1), np.linspace(0, h, m + 1))
    gx = gx.astype(int)  # np.int在numpy1.20已经被废弃掉了，因此此处用int或者np.int_，np.int32等
    gy = gy.astype(int)

    divided_grey = np.zeros([m, n, grid_h, grid_w], np.uint8)  # 四维张量
    if len(img.shape) == 3:
        divided = np.zeros([m, n, grid_h, grid_w, 3], np.uint8)  # 五维张量

        for i in range(m):
            for j in range(n):
                divided[i, j, ...] = resized[gy[i, j]:gy[i + 1, j + 1],
                                             gx[i, j]:gx[i + 1, j + 1], :]
                divided_grey[i, j, ...] = cv2.cvtColor(divided[i, j, ...], cv2.COLOR_BGR2GRAY)

        return divided, divided_grey
    else:
        for i in range(m):
            for j in range(n):
                divided_grey[i, j, ...] = resized[gy[i, j]:gy[i + 1, j + 1],
                                                  gx[i, j]:gx[i + 1, j + 1]]

        return divided_grey


def del_file(path):
    """
    删除指定目录下的文件，保留文件夹
    :param path:
    :return:
    """
    for i in os.listdir(path):
        # 如果存在文件夹进行递归
        if os.path.isdir(os.path.join(path, i)):
            del_file(os.path.join(path, i))
        # 如果是文件进行删除
        elif os.path.isfile:
            os.remove(os.path.join(path, i))


# 将divide分块的图片展示并保存，输入4维二值图矩阵或5维BGR矩阵
def show_blocks(divided):
    """
    Show and save the blocks you got through the "divide" function
    :param divided: 5-dim or 4-dim array
    :return: None
    """
    assert (len(divided.shape) == 5 or len(divided.shape) == 4)
    m, n = divided.shape[0], divided.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m, n, i * n + j + 1)
            if len(divided.shape) == 5:
                cv2.imwrite('./blocks/block_' + str(i * n + j + 1) + '.jpg', divided[i, j, :])
                plt.imshow(cv2.cvtColor(divided[i, j, :], cv2.COLOR_BGR2RGB))
            elif len(divided.shape) == 4:
                cv2.imwrite('./blocks/blockGrey_' + str(i * n + j + 1) + '.jpg', divided[i, j, :])
                plt.imshow(divided[i, j, :], cmap='Greys_r', vmin=0, vmax=255)
            plt.axis('off')
            plt.title('block' + str(i * n + j + 1), fontdict={'weight': 'normal', 'size': 10})
            plt.subplots_adjust(left=0.125,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.2,
                                hspace=0.35)

    plt.show()


# 将divide函数返回的blocks分别进行直方图均衡化和二值化并返回合成的4维二值图矩阵
def process_blocks(divided_grey, keyword='otsu', equalize=True, blur='after'):
    """
    Process the blocks respectively through histogram equalization and "otsu" method
    :param blur: choose between "before" and "after" to decide the blur being before or after equalization.
    :param equalize: bool
    :param divided_grey: 4-dim array
    :param keyword: 'otsu' or 'activate'
    :return: blocks_grey
    """
    assert (len(divided_grey.shape) == 4)
    blocks_grey = np.zeros(divided_grey.shape)
    m, n = divided_grey.shape[0], divided_grey.shape[1]
    for i in range(m):
        for j in range(n):
            if blur == 'before':  # 在直方图均衡化之前进行平滑滤波
                temp = cv2.blur(divided_grey[i, j, ...], (5, 5))
            else:
                temp = divided_grey[i, j, ...]
            if equalize:
                temp = cv2.equalizeHist(temp)
            if blur == 'after':  # 在直方图均衡化之后进行平滑滤波
                temp = cv2.blur(temp, (5, 5))
            blocks_grey[i, j, ...] = threshold(temp, keyword=keyword)

    return blocks_grey


# 将处理后的4维二值图矩阵进行合并，并返回最终合并后的图像
def avengers_assemble(blocks_grey):
    """
    Combine all the blocks
    :param blocks_grey: 4-dim array
    :return: assembled
    """
    assert (len(blocks_grey.shape) == 4)

    h = blocks_grey.shape[0] * blocks_grey.shape[2]
    w = blocks_grey.shape[1] * blocks_grey.shape[3]
    assembled = np.zeros((h, w))
    m, n = blocks_grey.shape[0], blocks_grey.shape[1]
    for i in range(m):
        for j in range(n):
            assembled[i * blocks_grey.shape[2]: (i + 1) * blocks_grey.shape[2],
                      j * blocks_grey.shape[3]: (j + 1) * blocks_grey.shape[3]] = blocks_grey[i, j, ...]
    return assembled


def process_run(img, keyword: str, row: int, colum: int):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    divided_grey = divide(img, row, colum)

    # 输出图像块，如果没有blocks目录则创建目录
    if os.path.exists('./blocks/'):
        del_file('./blocks/')
    else:
        os.makedirs('./blocks/')
    show_blocks(divided_grey)
    blocks_grey = process_blocks(divided_grey, keyword)
    assembled = avengers_assemble(blocks_grey)
    print(10 * '*' + 'Picture Processing Finished' + 10 * '*')

    return assembled


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    n = args.number
    k = args.keyword
    s = args.scale

    # 根据输入参数读图
    image = cv2.imread('./QRcode/' + str(n) + '.jpg')
    # 读取行列值
    r, c = int(s[0]), int(s[1])
    output = process_run(image, k, r, c)
    o = args.output
    cv2.imwrite('./QRcode/OUTPUT_EquAf' + str(o) + '_' + str(r) + 'x' + str(c) + '.jpg', output)
