# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年09月21日
"""

import cv2
import numpy as np


# TODO(机智的枫树): 本部分目的为锁定二维码范围并且切割放大
# 找出矩形的四个角点
def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# 将图片等比例缩小或放大
def resize(img, wth=None, ht=None, inter=cv2.INTER_AREA):
    (h, w) = img.shape[:2]
    if wth is None and ht is None:
        return img
    if wth is None:
        r = ht / float(h)
        dim = (int(w * r), ht)
    else:
        r = wth / float(w)
        dim = (wth, int(h * r))
    resized = cv2.resize(img, dim, interpolation=inter)

    return resized


if __name__ == '__main__':
    image = cv2.imread('./QRcode/3.png')
    height = 500
    ratio = image.shape[0] / float(height)
    orig = image.copy()
    image = resize(image, ht=height)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 应用高斯滤波去除部分噪音点
    edged = cv2.Canny(gray, 75, 200)

    print('step 1: edge detect')
    cv2.imshow('orig', image)
    cv2.imshow('edged', edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('step 2: get contour')

    cv2.imshow('outline', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
