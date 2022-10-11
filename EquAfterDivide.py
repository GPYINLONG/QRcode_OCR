# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年09月21日
"""


import cv2
import Process as Pro
import os


parser = Pro.get_parser()
args = parser.parse_args()
n = args.number
k = args.keyword
s = args.scale

# 根据输入参数读图
image = cv2.imread('./QRcode/' + str(n) + '.jpg')
# 读取行列值
r, c = int(s[0]), int(s[1])
# 高斯模糊
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)

print(10 * '*' + '1st STEP: Divide' + 10 * '*')
dividedGrey = Pro.divide(image, r, c)

# 输出图像块，如果没有blocks目录则创建目录
print(10 * '*' + '2nd STEP: Show the blocks' + 10 * '*')
if os.path.exists('./blocks/'):
    Pro.del_file('./blocks/')
else:
    os.makedirs('./blocks/')
Pro.show_blocks(dividedGrey)

print(10 * '*' + '3rd STEP: Process the blocks' + 10 * '*')
blocksGrey = Pro.process_blocks(dividedGrey, k)

print(10 * '*' + '4th STEP: Combine the blocks' + 10 * '*')
assembled = Pro.avengers_assemble(blocksGrey)

print(10 * '*' + '5th STEP: Output the final QR code' + 10 * '*')
o = args.output
cv2.imwrite('./QRcode/OUTPUT_EquAf' + str(o) + '_' + str(r) + 'x' + str(c) + '.jpg', assembled)
