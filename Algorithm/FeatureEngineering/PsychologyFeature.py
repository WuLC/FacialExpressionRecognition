# -*- coding: utf-8 -*-
# Created on Sat Jun 02 2018 10:26:30
# Author: WuLC
# EMail: liangchaowu5@gmail.com

from __future__ import division

import os
import math

import pickle
import dlib
import cv2
import fire
import numpy as np


PREDICTOR_PATH = 'F:/FacialExpressionRecognition/System/dlibmodel/shape_predictor_68_face_landmarks.dat'
DETECCTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)


def detect_landmarks(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = DETECCTOR(gray_img, 1)
    assert len(faces) == 1, 'detect no face or more than one face in image {0}'.format(img_path)
    d = faces[0]
    left, top, right, down = d.left(), d.top(), d.right(), d.bottom()
    print('detection face: left:{0} top:{1} right:{2} down:{3}'.format(left, top, right, down))
    landmarks = PREDICTOR(gray_img, d)
    coords = np.zeros((68, 2), dtype='int')
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

def extract_landmarks(src_dir, des_dir):
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    for img_name in os.listdir(src_dir):
        img_path = src_dir + img_name
        des_img_path = des_dir + img_name
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = DETECCTOR(gray_img, 1)
        assert len(faces) == 1, 'detect no face or more than one face in image {0}'.format(img_path)
        landmarks = PREDICTOR(gray_img, faces[0])
        landmarks_img = np.full(gray_img.shape, 0)
        for i in range(0, 68):
            coord = (landmarks.part(i).x, landmarks.part(i).y)
            cv2.circle(landmarks_img, center = coord, radius = 1, color = (255, 0, 0), thickness = 3)
        cv2.imwrite(des_img_path, landmarks_img)

def crop_face_with_landmarks(src_dir, des_dir, target_size = (224, 224)):
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    for img_name in os.listdir(src_dir):
        img_path = src_dir + img_name
        des_face_path = des_dir + img_name
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = DETECCTOR(gray_img, 1)
        assert len(faces) == 1, 'detect no face or more than one face in image {0}'.format(img_path)
        d = faces[0]
        left, top, right, down = d.left(), d.top(), d.right(), d.bottom()
        print('detection face: left:{0} top:{1} right:{2} down:{3}'.format(left, top, right, down))
        landmarks = PREDICTOR(gray_img, d)
        left_top = (landmarks.part(0).x, min(landmarks.part(19).y, landmarks.part(24).y))
        right_bottom = (landmarks.part(16).x, landmarks.part(8).y) 
        cropped_img = gray_img[left_top[1] : right_bottom[1], left_top[0] : right_bottom[0]]
        resized_img = cv2.resize(cropped_img, target_size)
        cv2.imwrite(des_face_path, resized_img)

def distance(p1, p2):
    return pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2)

def visualize_landmark_change(img_dir, n = 20):
    coords = []
    for img_name in sorted(os.listdir(img_dir)):
        img_path = img_dir + img_name
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = DETECCTOR(gray_img, 1)
        assert len(faces) == 1, 'detect no face or more than one face in image {0}'.format(img_path)
        landmarks = PREDICTOR(gray_img, faces[0])
        coords.append([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 68)])
    
    # calculate changes, scale and visualize
    diff = []
    min_val, max_val = 10**9, 0
    for i in range(1, len(coords)):
        curr = []
        for j in range(len(coords[i])):
            delta_x = coords[i][j][0] - coords[i-1][j][0]
            delta_y = coords[i][j][1] - coords[i-1][j][1]
            distance = delta_x**2 + delta_y**2
            d = math.sqrt(distance)
            tmp = [0] * n
            if d > 0:
                radian = math.acos(delta_y/d)
                if delta_x < 0:
                    radian = 2 * math.pi - radian
                idx = (int((radian - math.pi/n) / (2*math.pi/n)) + 1) % n
                assert 0 <= idx < n, "area not legal"
                tmp[idx] = distance
            curr += tmp
            min_val, max_val = min(min_val, distance), max(max_val, distance)
        diff.append(curr)

    # scale to 0~255
    for i in range(len(diff)):
        for j in range(len(diff[i])):
            diff[i][j] = int(255.0*(diff[i][j] - min_val)/(max_val - min_val))
    img = np.array(diff)
    cv2.imwrite(img_dir + 'landmark_diff.png', img)


def face_total_distance(coords):
    return sum(distance(coords[33],  coords[i]) for i in range(68))


def get_eye_feature(coords):
    feature = []
    # 眉毛到眉心距离
    for i in range(17, 27):
        feature.append(distance(coords[i], coords[27]))
    
    # 眉间距离
    for i in range(17, 22):
        feature.append(distance(coords[i], coords[43-i]))

    # 单侧眉毛距离
    for i, j in ((18, 20), (17, 21), (23, 25), (22, 26)):
        feature.append(distance(coords[i], coords[j]))

    # 眼睛上下左右距离
    for i, j in ((37, 41), (38, 40), (36, 39), (43, 47), (44, 46), (42, 45)):
        feature.append(distance(coords[i], coords[j]))
    
    # 眉毛到下眼睑的距离
    for i, j in ((18, 36), (19, 41), (20, 40), (21, 39), (22, 42), (23, 47), (24, 46), (25, 45)):
        feature.append(distance(coords[i], coords[j]))

    # 两眼的距离
    for i, j in ((39, 42), (38, 43), (40, 47), (36, 45)):
        feature.append(distance(coords[i], coords[j]))
    
    # 眉毛到眉心的距离
    for i, j in ((27, 21), (27, 22), (27, 28)):
        feature.append(distance(coords[i], coords[j]))

    # 眼睛中心到眉毛和鼻子形成的直线的距离
    left_eye_enter, right_eye_center = np.zeros((2,1)), np.zeros((2,1))
    for left in [37, 38, 40, 41]:
        left_eye_enter[0] += coords[left][0]/4.0
        left_eye_enter[1] += coords[left][1]/4.0
    for right in [43, 44, 46, 47]:
        right_eye_center[0] += coords[right][0]/4.0
        right_eye_center[1] += coords[right][1]/4.0
    from numpy.linalg import norm
    feature.append(norm(np.cross(coords[19] - coords[33], coords[33] - left_eye_enter))/norm(coords[19] - coords[33]))
    feature.append(norm(np.cross(coords[24] - coords[33], coords[33] - right_eye_center))/norm(coords[24] - coords[33]))
    return feature


def get_mouth_feature(coords):
    feature= []
    # 上下左右距离
    for i, j in ((61, 67), (62, 66), (63, 65), (60, 64), (48, 54), (49, 59), (50, 58), (51, 57), (52, 56), (53, 55)):
        feature.append(distance(coords[i], coords[j]))
    return feature


def get_cheek_feature(coords):
    feature = []
    # 脸颊左右距离
    for i, j in ((0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9)):
        feature.append(distance(coords[i], coords[j]))
    return feature


def get_curve_feature(coords):
    def get_circle_curvature(p1, p2, p3):
        """return curvature of the circle constructed by 3 points"""
        x, y, z = complex(p1[0], p1[1]), complex(p2[0], p2[1]), complex(p3[0], p3[1]),
        w = z-x
        w /= y-x
        if abs(w.imag-x) < 1e-5:
            return 0
        try:
            c = (x-y)*(w-abs(w)**2)/(2j*(w.imag-x))
        except Exception:
            print(w.imag-x)
        center = (c.real, c.imag)
        radius = abs(c+x)
        return 1.0/radius
    feature = []
    # 眉毛取三点形成的圆的曲率
    for i in range(17, 20):
        feature.append(get_circle_curvature(coords[i], coords[i+1], coords[i+2]))
    feature.append(get_circle_curvature(coords[17], coords[19], coords[21]))
    for i in range(22, 25):
        feature.append(get_circle_curvature(coords[i], coords[i+1], coords[i+2]))
    feature.append(get_circle_curvature(coords[22], coords[24], coords[26]))

    # 眼睛取三点形成的圆的曲率
    for i, j, k in ((36, 37, 38), (37, 38, 39), (36, 41, 40), (39, 40, 41)):
        feature.append(get_circle_curvature(coords[i], coords[j], coords[k]))
    for i, j, k in ((42, 43, 44), (43, 44, 45), (42, 47, 46), (47, 46, 45)):
        feature.append(get_circle_curvature(coords[i], coords[j], coords[k]))
    
    # 鼻子取三点形成的圆的曲率
    for i in range(31, 34):
       feature.append(get_circle_curvature(coords[i], coords[i+1], coords[i+2]))
    feature.append(get_circle_curvature(coords[31], coords[33], coords[35]))
    
    # 脸颊取三点形成的圆的曲率
    for i in range(0, 15):
        feature.append(get_circle_curvature(coords[i], coords[i+1], coords[i+2]))

    # 嘴巴取三点形成的圆的曲率
    for i, j, k in ((48, 59, 58), (59, 58, 57), (58, 57, 56), (57, 56, 55), (56, 55, 54), (61, 62, 63), (65, 66, 67), (48, 49, 50), (52, 53, 54)):
        feature.append(get_circle_curvature(coords[i], coords[j], coords[k]))
    return feature


def get_wrinkle_feature(img_path):
    """Get wrinkle feature with Gabor filters"""
    def get_forehead_img(img_path):
        img = cv2.imread(img_path)[:, :, 0]
        coords = detect_landmarks(img_path)
        forehead_height = coords[30][1] - coords[27][1]
        left = coords[18][0]
        top = coords[19][1] - forehead_height
        right = coords[25][0]
        down  = coords[19][1]
        # print(left, right, top, down, img.shape)
        forehead_img = img[top:down, left:right]
        # print(forehead_img.shape)
        return forehead_img

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

    def connected_blocks(img, m, n, area_threshold=150):
        count = 0
        for i in range(m):
            for j in range(n):
                if img[i][j] > 0 and dfs(img, m, n, i, j) > area_threshold:
                    count += 1
        return count
    

    # process wrinkle image
    filters = build_filters()
    processed_image = process(get_forehead_img(img_path), filters)

    # extract feature from processed image
    feature = []
    m, n = processed_image.shape
    feature.append(wrinkle_prop(processed_image))
    #feature.append(connected_blocks(processed_image.tolist(), m, n))
    return feature


def generate_data():
    img_dir = 'E:/FaceExpression/TrainSet/CK+/10_fold_original/'
    pkl_file = './psychology_feature.pkl'
    X, Y = [], []
    for i in range(1, 11):
        fold_dir = img_dir + 'g{0}/'.format(i)
        fold_x, fold_y = [], []
        for sample in os.listdir(fold_dir):
            label = int(sample.split('_')[0]) - 1
            sample_path = fold_dir + '{0}/'.format(sample)
            img_path = sample_path + sorted(os.listdir(sample_path))[-1]
            coords = detect_landmarks(img_path)
            face_size = face_total_distance(coords)
            feature = get_eye_feature(coords)
            feature += get_mouth_feature(coords)
            feature += get_cheek_feature(coords)
            #feature = [val/face_size for val in feature]
            feature += get_curve_feature(coords)
            feature += get_wrinkle_feature(img_path)
            fold_x.append(feature)
            fold_y.append(label)
        X.append(fold_x)
        Y.append(fold_y)
    with open(pkl_file, mode='wb') as wf:
        pickle.dump([X, Y], wf)
    print('====================finish dumping data======================')

if __name__ == '__main__':
    fire.Fire()