# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年09月21日
"""

import cv2
import numpy as np


# TODO(机智的枫树)：本部分目的为锁定二维码范围并且切割放大
def preprocess(img, show=True):
    """
    Preprocess the image input image
    :param img: 
    :param show: Bool
    :return: th
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    """
    This function aims at finding and printing contours of the three location signs. Return contours list
    :param img: Gray image
    :param draw: Bool
    :return: List
    """
    cnts2 = []
    drawing = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cnts, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                    # drawing = cv2.drawContours(drawing, cnts, i, (0, 0, 255), 3)
                    # 记录搜索到的两个子轮廓并存储其编号
                    cnts2.append(np.squeeze(cnts[i]))
                    # cnts2.append(cnts[temp1])
                    # cnts2.append(cnts[temp2])
    drawing = cv2.drawContours(drawing, cnts2, -1, (0, 0, 255), 3)
    assert len(cnts2) == 3
    cv2.imshow('Contours', drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if draw:
        cv2.imwrite('./QRcode/Contours.jpg', drawing)
    print(cnts2[0].shape)
    print(cnts2[1].shape)
    print(cnts2[2].shape)
    return cnts2


def compute_center(cnts: list):
    # 在矩形框上每隔1/4个周长取一个点，计算这四个点的平均坐标作为矩形框的中心坐标
    assert type(cnts[0]) == np.ndarray
    l = len(cnts)
    centers = np.zeros((l, 2), dtype=np.float32)
    for i in range(l):
        n = len(cnts[i]) - 1
        centers[i] = (cnts[i][n] + cnts[i][int(n * 3 / 4)] + cnts[i][int(n / 2)] + cnts[i][int(n / 4)]) / 4.0

    return centers


def signs_order(centers: np.ndarray):
    assert centers.shape == (3, 2)
    angle = np.zeros(3, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    a = centers[1] - centers[0]
    b = centers[2] - centers[0]
    c = centers[1] - centers[2]
    angle[0] = 180 / np.pi * np.arccos((np.dot(a, a.T) + np.dot(b, b.T) - np.dot(c, c.T)) / (2 * np.sqrt(np.dot(a, a.T))
                                                                                             * np.sqrt(np.dot(b, b.T))))

    a = centers[2] - centers[1]
    b = centers[0] - centers[1]
    c = centers[2] - centers[0]
    angle[1] = 180 / np.pi * np.arccos((np.dot(a, a.T) + np.dot(b, b.T) - np.dot(c, c.T)) / (2 * np.sqrt(np.dot(a, a.T))
                                                                                             * np.sqrt(np.dot(b, b.T))))

    a = centers[0] - centers[2]
    b = centers[1] - centers[2]
    c = centers[0] - centers[1]
    angle[2] = 180 / np.pi * np.arccos((np.dot(a, a.T) + np.dot(b, b.T) - np.dot(c, c.T)) / (2 * np.sqrt(np.dot(a, a.T))
                                                                                             * np.sqrt(np.dot(b, b.T))))

    max_index = np.argmax(angle)
    rect[0] = centers[max_index]  # 角度最大的点即为左上点
    centers = np.delete(centers, max_index, axis=0)
    # 判断最大角标志点位于另外两个标志点中点的左边还是右边，如果是左边则上边的点为右上点，如果位于右边则下边的点为右上点
    median = centers.sum(axis=0) / 2.0
    if rect[0, 0] < median[0]:
        rect[1] = centers[np.argmin(centers[:, 1])]
        rect[2] = centers[np.argmax(centers[:, 1])]
    else:
        rect[1] = centers[np.argmax(centers[:, 1])]
        rect[2] = centers[np.argmin(centers[:, 1])]
    rect[3] = rect[1] + rect[2] - rect[0]
    return rect, max_index


def compute_rdc(rect: np.ndarray, cnts: list, index):
    """
    Compute the right down contour.
    :param rect: np.ndarray
    :param cnts: list
    :return: cnts
    """
    rdc = cnts[index] + (rect[3] - rect[0])
    cnts.append(rdc)
    return cnts


# TODO(机智的枫树)：后续需要把根据矩形框的角度把矩形框四个角成比例向外侧移动至括入完整二维码图形
def extract_min_rect(cnts: list, img: np.ndarray):
    len0 = len(cnts[0])
    len1 = len(cnts[1])
    len2 = len(cnts[2])
    len3 = len(cnts[3])
    temp = np.zeros((len0 + len1 + len2 + len3, 2), dtype=np.float32)
    temp[0: len0, :] = cnts[0]
    temp[len0: len0 + len1, :] = cnts[1]
    temp[len0 + len1: len0 + len1 + len2, :] = cnts[2]
    temp[len0 + len1 + len2: len0 + len1 + len2 + len3, :] = cnts[3]

    rect = cv2.minAreaRect(temp)
    box = np.int_(cv2.boxPoints(rect))
    cv2.imwrite('./QRcode/box.jpg', cv2.drawContours(img, [box], -1, (0, 0, 255), 2))


def four_point_transform(img, rect):  # 进行图片比例变换
    # 获取输入坐标点
    (tl, tr, bl, br) = rect  # top left, top right, bottom right, bottom left
    # 计算输入的w和h值
    width_a = np.sqrt(((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2))

    height_a = np.sqrt(((tr[0] - bl[0]) ** 2 + (tr[1] - bl[1]) ** 2))
    height_b = np.sqrt(((tl[0] - br[0]) ** 2 + (tl[1] - br[1]) ** 2))
    max_len = max(int(height_a), int(height_b), int(width_a), int(width_b))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [max_len - 1, 0],
        [0, max_len - 1],
        [max_len - 1, max_len - 1]
    ], dtype='float32')
    # 计算变换矩阵
    # cv2.getPerspectiveTransform(src, dst) → retval，将成像投影到一个新的视平面
    # 参数：src：源图像中待测矩形的四点坐标；sdt：目标图像中矩形的四点坐标；
    # 返回由源图像中矩形到目标图像矩形变换的矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # cv2.warpPerspective(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) --> dst，
    # 透视变换函数，可保持直线不变形，但是平行线可能不再平行
    # 参数：src：输入图像；dst：输出图像；M：变换矩阵；dsize：变换后输出图像尺寸；flag：插值方法；borderMode：边界像素外扩方式；borderValue：边界像素插值，默认用0填充
    warped = cv2.warpPerspective(img, M, (max_len, max_len))
    # 返回变换后结果
    return warped


if __name__ == '__main__':
    image = cv2.imread('./QRcode/2.jpg')
    orig = image.copy()
    th1 = preprocess(orig)
    cnts = find_contour(th1)
    centers = compute_center(cnts)
    rect, index = signs_order(centers)
    cnts = compute_rdc(rect, cnts, index)
    extract_min_rect(cnts, orig)
    warped = four_point_transform(image, rect)
    cv2.imshow('warped', warped)
    cv2.imwrite('./QRcode/warped.jpg', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

