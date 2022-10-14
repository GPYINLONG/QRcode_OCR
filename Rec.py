# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年10月14日
"""

import pyzbar.pyzbar as pyz
import cv2


def rec_run(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    barcode = pyz.decode(img)
    for result in barcode:
        result = result.data.decode('utf-8')
        print('Result:', result)
    print(10 * '*' + 'Recognition Success' + 10 * '*')


if __name__ == '__main__':
    image = cv2.imread('./QRcode/warped.jpg')
    rec_run(image)

