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
                        help='Please input the number of the picture you wanna process.')
    parser.add_argument('-k', '-keyword', choices=['otsu', 'adaptive'], required=True,
                        help=r'Please choose the algorithm of the threshold function between \'otsu\' and \'adaptive\'.')

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


if __name__ == '__main__':
    img = cv2.imread('./QRcode/1.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
