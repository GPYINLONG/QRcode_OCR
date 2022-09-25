# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年09月21日
"""

import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '-number', required=True, type=int,
                        help='Input the number of the picture you wanna process.')
    parser.add_argument('-k', '-keyword', choices=['otsu', 'adaptive'], required=True,
                        help=r'Choose the algorithm of the threshold function between \'otsu\' and \'adaptive\'.')
    parser.add_argument('-s', '-scale', required=True,
                        help='Input a (m, n) tuple to divide the picture into m × n blocks.')

    return parser


def threshold(img, keyword='otsu', max_val=0, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
              threshold_type=cv2.THRESH_BINARY, block_size=3, c=2):
    if keyword == 'adaptive':
        th = cv2.adaptiveThreshold(img, max_val, adaptive_method, threshold_type, block_size, c)
        return th
    elif keyword == 'otsu':
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    else:
        print('Keyword Error!')


# 将图片分为m×n块
def divide(img, m, n):
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / m)
    grid_w = int(w * 1.0 / n)

    # 计算满足整除关系的高与宽
    h = grid_h * m
    w = grid_w * n

    # 进行图像缩放
    resized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.linspace(0, w, n + 1), np.linspace(0, h, m + 1))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)

    divided = np.zeros([m, n, grid_h, grid_w, 3], np.uint8)  # 五维张量

    for i in range(m):
        for j in range(n):
            divided[i, j, ...] = resized[gy[i, j]:gy[i + 1, j + 1],
                                         gx[i, j]:gx[i + 1, j + 1], :]

    return divided


def show_blocks(divided):
    m, n = divided.shape[0], divided.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m, n, i * n + j + 1)
            cv2.imwrite('./blocks/block'+str(i * n + j + 1)+'.jpg', divided[i, j, :])
            plt.imshow(cv2.cvtColor(divided[i, j, :], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('block'+str(i * n + j + 1), fontdict={'weight': 'normal', 'size': 10})
            plt.subplots_adjust(left=0.125,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.2,
                                hspace=0.35)

    plt.show()


if __name__ == '__main__':
    img = cv2.imread('./QRcode/1.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fig1 = plt.figure('Origin')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    thr = threshold(gray)
    fig2 = plt.figure('OTSU')
    plt.axis('off')
    plt.imshow(thr, cmap='Greys_r', vmin=0, vmax=255)

    divide_img = divide(img, 4, 4)
    fig2 = plt.figure('Blocks')
    show_blocks(divide_img)

