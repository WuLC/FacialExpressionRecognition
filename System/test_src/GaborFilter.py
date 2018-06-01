# -*- coding: utf-8 -*-
# Created on Thu May 31 2018 19:17:48
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import numpy as np
import cv2 

def build_filters():
    filters = []
    ksize = 40
    for theta in [np.pi/2]: #np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel(ksize = (40, 40), 
                                  sigma = 8.0, 
                                  theta = np.pi/2, 
                                  lambd = 12.0, 
                                  gamma = 0.3, 
                                  psi = 0, 
                                  ktype = cv2.CV_32F)
        #kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def wrinkle_prop(img):
    m, n  = img.shape
    white_count = 0
    for i in range(m):
        for j in range(n):
            if img[i][j] != 0:
                white_count += 1
    return white_count/(m*n)

def connected_blocks(img, m, n):
    count = 0
    area_threshold = 150
    for i in range(m):
        for j in range(n):
            if img[i][j] > 0 and dfs(img, m, n, i, j) > area_threshold:
                count += 1
    return count

def dfs(img, m, n, i, j):
    """return area of connected block"""
    img[i][j] = -1
    area = 1
    if i - 1 >= 0 and img[i-1][j] > 0:
        area += dfs(img, m, n, i-1, j)
    if i + 1 < m and img[i+1][j] > 0:
        area += dfs(img, m, n, i+1, j)
    if j - 1 >= 0 and img[i][j-1] > 0:
        area += dfs(img, m, n, i, j-1)
    if j + 1 < n and img[i][j+1] > 0:
        area += dfs(img, m, n, i, j+1)
    return area
    
def main():
    img_dir = 'E:/FaceExpression/TrainSet/CK+/10_fold_original/'
    img_name = 'g1/3_S005_001/S005_001_00000002.png'
    img_name = 'g2/4_S050_001/S050_001_00000009.png'
    im1 = 'test3.png'
    im2 = 'test4.png'
    img1 = cv2.imread(img_dir+im1)[:,:,0]
    img2 = cv2.imread(img_dir+im2)[:,:,0]
    filters = build_filters()
    res1 = process(img1, filters)
    res2 = process(img2, filters)
    print(wrinkle_prop(res1))
    print(wrinkle_prop(res2))
    
    # count connected blocks
    m, n = res1.shape
    print(connected_blocks(res1.tolist(), m, n))
    print(connected_blocks(res2.tolist(), m, n))
    cv2.imshow(img_name, np.hstack((res1, res2)))
    cv2.waitKey(0)
    #cv2.destroyAllWindows()  

if __name__ == '__main__':
    main()