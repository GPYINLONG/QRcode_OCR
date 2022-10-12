# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年09月21日
"""

import cv2
import numpy as np


# TODO(机智的枫树)：本部分目的为锁定二维码范围并且切割放大
# 找出二维码的三个角的定位角点


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
    print(len(cnts2))
    print(cnts2[0].shape)
    return cnts2


def corners_order(cnts: list):
    """
    Input a list containing arrays and return a list containing 3 "ndarray" corners, with their 4 corners in order of
    left up, right up, left down and right down
    :param cnts: List
    :return: List
    """
    # TODO(机智的枫树)：出现问题，现有算法没办法从上百个点中挑出正好位于四个角的点，需要做改进，将一个标志的所有点做一个坐标平均得到中心坐标，将
    #  中心坐标进行排序得到三个标志的顺序，而后再用其中一个中心坐标减去其轨迹组内任一点坐标得到Δx和Δy，再分别根据四个标志的顺序±Δx和Δy以得到外角点坐标
    rects = []
    for pts in cnts:
        assert type(pts) == np.ndarray
        temp = np.zeros((4, 2), dtype='float32')
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)  # np.diff(a, n) x为输入的数组，n为差分的次数，执行后一个元素减前一个元素
        # 计算左上和右下
        temp[0] = pts[np.argmin(s)]  # np.argmin无下表时默认将数组展平并输出最小元素的下标
        temp[3] = pts[np.argmax(s)]

        # 计算右上和左下
        temp[1] = pts[np.argmin(diff)]
        temp[2] = pts[np.argmax(diff)]

        rects.append(temp)
        print('rects:', rects)

    return rects


# TODO(机智的枫树)：后续需要把根据矩形框的角度把矩形框四个角成比例向外侧移动至括入完整二维码图形
def extract_min_rect(cnts: list, img: np.ndarray):
    temp = np.zeros((len(cnts[0]) + len(cnts[1]) + len(cnts[2]), 2), dtype=np.float32)
    print(temp.shape)
    temp[0: len(cnts[0]), :] = cnts[0]
    temp[len(cnts[0]): len(cnts[0]) + len(cnts[1]), :] = cnts[1]
    temp[len(cnts[0]) + len(cnts[1]): len(cnts[0]) + len(cnts[1]) + len(cnts[2]), :] = cnts[2]

    rect = cv2.minAreaRect(temp)
    box = np.int_(cv2.boxPoints(rect))
    # cv2.imshow('box', cv2.drawContours(img, [box], -1, 255, 2))
    cv2.imwrite('./QRcode/box.jpg', cv2.drawContours(img, [box], -1, 255, 2))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def signs_order(rects: list):
    # signs_order函数输入corners_order的返回列表并将其中三个标志的各4组行列坐标分别加和除以4代表他们的中心坐标，
    #  而后继续通过corners_order中的算法判断标志顺序位置，最后计算第四个标志的外角点位置
    assert type(rects[0]) == np.ndarray
    rect = np.zeros((4, 2), dtype=np.float32)
    pts = np.zeros((3, 2), dtype=np.float32)
    # 将三个标志的各4组角点行列坐标分别加和除以4以代表他们的中心坐标
    for i in range(len(rects)):
        pts[i] = rects[i].sum(axis=0) / 4
    print(pts)
    print(rects)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    # 根据三个标志的中心坐标找到对应左上、右上和左下的标志外角点
    rect[0] = rects[np.argmax(s)][0]
    rect[1] = rects[np.argmin(diff)][1]
    rect[2] = rects[np.argmax(diff)][2]
    rect[3, 0] = rect[2, 0] + (rect[1, 0] - rect[0, 0])
    rect[3, 1] = rect[2, 1] + (rect[1, 1] - rect[0, 1])
    print(rect)

    return rect


def four_point_transform(img, rect):  # 进行图片比例变换
    # 获取输入坐标点
    (tl, tr, bl, br) = rect  # top left, top right, bottom right, bottom left
    # 计算输入的w和h值
    widthA = np.sqrt(((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - bl[0]) ** 2 + (tr[1] - bl[1]) ** 2))
    heightB = np.sqrt(((tl[0] - br[0]) ** 2 + (tl[1] - br[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')
    # 计算变换矩阵
    # cv2.getPerspectiveTransform(src, dst) → retval，将成像投影到一个新的视平面
    # 参数：src：源图像中待测矩形的四点坐标；sdt：目标图像中矩形的四点坐标；
    # 返回由源图像中矩形到目标图像矩形变换的矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # cv2.warpPerspective(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) --> dst，透视变换函数，可保持直线不变形，但是平行线可能不再平行
    # 参数：src：输入图像；dst：输出图像；M：变换矩阵；dsize：变换后输出图像尺寸；flag：插值方法；borderMode：边界像素外扩方式；borderValue：边界像素插值，默认用0填充
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # 返回变换后结果
    return warped


if __name__ == '__main__':
    image = cv2.imread('./QRcode/2.jpg')
    orig = image.copy()
    th1 = preprocess(orig)
    cnts = find_contour(th1)
    extract_min_rect(cnts, orig)
    rects = corners_order(cnts)
    rect = signs_order(rects)
    warped = four_point_transform(image, rect)
    cv2.imshow('warped', warped)
    cv2.imwrite('./QRcode/warped.jpg', warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

