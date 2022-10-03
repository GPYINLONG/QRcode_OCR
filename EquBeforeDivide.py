# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年09月26日
"""

import cv2
import Pre
import os


parser = Pre.get_parser()
args = parser.parse_args()
n = args.number
k = args.keyword
s = args.scale

# 根据输入参数读图
image = cv2.imread('./QRcode/'+str(n)+'.jpg')
# 读取行列值
r, c = int(s[0]), int(s[1])

print(10 * '*' + '1st STEP: Histogram' + 10 * '*')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.equalizeHist(image)

print(10 * '*' + '2nd STEP: Divide' + 10 * '*')
dividedGrey = Pre.divide(image, r, c)

# 输出图像块，如果没有blocks目录则创建目录
print(10 * '*' + '3rd STEP: Show the blocks' + 10 * '*')
if os.path.exists('./blocks/'):
    Pre.del_file('./blocks/')
else:
    os.makedirs('./blocks/')
    Pre.show_blocks(dividedGrey)

print(10 * '*' + '4th STEP: Process the blocks' + 10 * '*')
blocksGrey = Pre.process_blocks(dividedGrey, k, equalize=False)

print(10 * '*' + '5th STEP: Combine the blocks' + 10 * '*')
assembled = Pre.avengers_assemble(blocksGrey)

print(10 * '*' + '6th STEP: Output the final QR code' + 10 * '*')
o = args.output
cv2.imwrite('./QRcode/OUTPUT_EquBf' + str(o) + '_' + str(r) + 'x' + str(c) + '.jpg', assembled)
