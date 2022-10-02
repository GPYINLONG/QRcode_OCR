# QRcode_OCR
## 图像处理课程作业
本项目是基于西北农林科技大学屈卫锋等的文章《光照不均QR二维码图像的高效处理方法研究》所进行的代码复现，并在此基础上进行进一步的实验，目的之一是完成田劲东老师的图像处理课程作业，再就是锻炼自己的代码能力与创新思维能力和OpenCV的熟练度。
### 目标

- [x] 定位二维码位置并进行图像裁剪
- [ ] 复现直方图均衡化后Otsu分割方法并复原二维码
- [x] 对二维码中的信息进行识别

### NEXT STEP
#### 关于目标一
- [x] 现拟用OpenCV对二维码三个角点进行检测，实现识别并定位
#### 关于目标二
- [x] 调整二值化方法与直方图均衡化方法
- [ ] 分割更小图片
- [ ] 添加高频滤波组件

<font color=DeepSkyBlue>*实验证明分割更小图片并分别进行Histogram Equalization后虽然能起到更好的恢复效果，但是会导致图片更容易受到噪点的影响。添加高斯滤波后效果仍然不理想。*</font>

### 先均衡化后分割
调整图像分割和直方图均衡化的先后顺序，先对整幅图像进行直方图均衡化后对输出图像进行块分割。

<font color=DeepSkyBlue>*实验证明先均衡化后分割二值化则导致分割步骤毫无作用，该方法被证明无效。*</font>

### Install Requirements
跑 `pip install -r requirements.txt` 以安装所有需要的包。

## References
> [1] 屈卫锋, 徐越, 牛磊磊,等. 光照不均QR二维码图像的高效处理方法研究[J]. 软件, 2015(6):6.
> 
> [2] OTSU算法（大津法—最大类间方差法）原理及实现 https://blog.csdn.net/weixin_40647819/article/details/90179953
>
> [3] python-opencv图像切块 https://blog.csdn.net/hymn1993/article/details/122789718
> 
> [4] 二维码的特征定位和信息识别 https://blog.csdn.net/iamqianrenzhan/article/details/79117119
> 
> [5] OpenCV二维码检测与定位 http://www.manongjc.com/detail/50-whhxujelpahjvil.html