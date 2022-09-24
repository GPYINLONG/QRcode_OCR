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


if __name__ == '__main__':
    img = cv2.imread('./QRcode/1.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
