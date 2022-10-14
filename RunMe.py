# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年10月14日
"""

import cv2
import Process as Pro
import Loc
import Rec

parser = Pro.get_parser()
args = parser.parse_args()
n = args.number
k = args.keyword
s = args.scale

# 根据输入参数读图
image = cv2.imread('./QRcode/' + str(n) + '.jpg')

warped = Loc.loc_run(image)

r, c = int(s[0]), int(s[1])
output = Pro.process_run(warped, k, r, c)
o = args.output
cv2.imwrite('./QRcode/OUTPUT_EquAf' + str(o) + '_' + str(r) + 'x' + str(c) + '.jpg', output)

Rec.rec_run(output)
