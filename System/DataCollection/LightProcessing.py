# -*- coding: utf-8 -*-
# Created on Wed Dec 06 2017 16:29:42
# Author: WuLC
# EMail: liangchaowu5@gmail.com

"""
import cv2
img = cv2.imread('./001.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[:,:,2] += 255
img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('./processed_001.jpg', img)


import os
import numpy as np
import cv2
if __name__ == '__main__':
        img = cv2.imread('./002.jpg', 0)
        img = np.array(img)
        mean = np.mean(img)
        img = img - mean
        img = img*0.7 + mean*1.2 #修对比度和亮度
        img = img/255 #非常关键，没有会白屏
        cv2.imshow('pic',img)
        cv2.waitKey()
"""
import cv2
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt

g1 = 2 #调暗
g2 = 0.4 #调亮

np_img = cv2.imread('./001.jpg')
image = img_as_float(np_img)
gam1= exposure.adjust_gamma(image, g1)
gam2= exposure.adjust_gamma(image, g2)   
plt.figure('adjust_gamma')

plt.subplot(211)
plt.title('origin image')
plt.imshow(image,plt.cm.gray)
plt.axis('off')

# plt.subplot(312)
# plt.title('gamma=2')
# plt.imshow(gam1,plt.cm.gray)
# plt.axis('off')

plt.subplot(212)
plt.title('gamma={0}'.format(g2))
plt.imshow(gam2,plt.cm.gray)
plt.axis('off')

plt.show()