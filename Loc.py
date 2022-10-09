# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年09月21日
"""

import cv2
import numpy as np


# TODO(机智的枫树): 本部分目的为锁定二维码范围并且切割放大
# 找出二维码的三个角的定位角点
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


def preprocess(img, show=True):
    """
    Preprocess the image input image
    :param img: 
    :param show: Bool
    :return: th
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))  # 应用平滑滤波去除部分噪音点
    gray_equ = cv2.equalizeHist(gray)
    # ret, th = cv2.threshold(gray_equ, 112, 255, cv2.THRESH_BINARY)
    ret, th = cv2.threshold(gray_equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # otsu效果更好

    if show:
        cv2.namedWindow('gray')
        cv2.namedWindow('threshold')
        cv2.imshow('gray', gray_equ)
        cv2.imshow('threshold', th)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return th


def find_contour(img, draw=True):
    cnts2 = []
    drawing = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cnts, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # RETR_TREE以树形结构组织输出，使得hierarchy的四列分别对应下一个轮廓编号、上一个轮廓编号、子轮廓编号、父轮廓编号，该值为负数表示没有对应项。
    for i in range(len(cnts)):
        if hierarchy[0, i, 2] == -1:
            continue
        else:
            temp1 = hierarchy[0, i, 2]  # 第一个子轮廓的索引
            if hierarchy[0, temp1, 2] == -1:
                continue
            else:
                temp2 = hierarchy[0, temp1, 2]  # 第二个子轮廓的索引
                len1 = cv2.arcLength(cnts[i], closed=True)
                len2 = cv2.arcLength(cnts[temp1], closed=True)
                len3 = cv2.arcLength(cnts[temp2], closed=True)
                if abs(len1 / len2 - 2) <= 1 and abs(len2 / len3 - 2) <= 1:  # 筛选满足长度比例的轮廓
                    drawing = cv2.drawContours(drawing, cnts, i, (0, 0, 255), 3)
                    # 记录搜索到的两个子轮廓并存储其编号
                    cnts2.append(cnts[i])
                    cnts2.append(cnts[temp1])
                    cnts2.append(cnts[temp2])
    # drawing = cv2.drawContours(drawing, cnts2, -1, (0, 0, 255), 3)
    cv2.imshow('Contours', drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if draw:
        cv2.imwrite('./QRcode/Contours.jpg', drawing)
    print(cnts2)
    print(len(cnts2))


if __name__ == '__main__':
    image = cv2.imread('./QRcode/2.jpg')
    orig = image.copy()
    th1 = preprocess(orig)
    find_contour(th1)
    height = 500
    ratio = image.shape[0] / float(height)

    image = resize(image, ht=height)
