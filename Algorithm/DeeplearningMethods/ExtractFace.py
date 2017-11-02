# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-07-27 10:43:58
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-11 21:19:02

import os 
import sys

import cv2
import pickle 
import numpy as np

from keras.preprocessing import image


def extract_face(input_image, output_image):
    cascPath = "./haarcascade_frontalface_default.xml"
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(input_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    if len(faces) == 0:
        print("Found no faces in image {0}".format(input_image))
        return

    # save the image
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = image[y : y+h, x : x+w, :]
        cv2.imwrite(output_image, face)
        # print(face.shape)


def ck_batch_face_extraction():
    base_dir = './cohn-kanade-images-10-groups/'
    dest_dir = './cohn-kanade-images-10-groups_faces/'
    label_file = 'CKplus10groups.txt'
    target_file = base_dir+label_file
    count = 0
    with open(target_file, 'r') as f:
        for line in f:
            count += 1
            if count % 100 == 0:
                print('finish {0} images'.format(count))
            file_path, label = line.strip().replace('\\', '/').split()
            source_path = base_dir + file_path
            dest_path = dest_dir + file_path
            dest_final_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_final_dir):
                os.makedirs(dest_final_dir)
            extract_face(source_path, dest_path)
        

def kdef_batch_face_extraction():
    base_dir = './KDEF_G/'
    dest_dir = './KDEF_G_face/'
    label_file = 'label.txt'
    target_file = base_dir+label_file
    count = 0
    with open(target_file, 'r') as f:
        for line in f:
            count += 1
            if count % 100 == 0:
                print('finish {0} images'.format(count))
            file_name, label, group = line.strip().split()
            source_path = base_dir+'Group{0}/'.format(group)+file_name
            dest_path = dest_dir + 'Group{0}/'.format(group)+file_name
            dest_final_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_final_dir):
                os.makedirs(dest_final_dir)
            extract_face(source_path, dest_path)


def ck_image_path_and_label():
    base_dir = './cohn-kanade-images-10-groups_faces/'
    label_file = 'CKplus10groups.txt'
    target_file = base_dir + label_file
    all_folds = []
    curr_fold = []
    previous = None
    with open(target_file, 'r') as f:
        for line in f:
            file_path, label = line.strip().replace('\\', '/').split()
            # remove the second category and change the seventh to the second
            if label == '2': 
                continue
            if label == '7':
                label = '2'
            if previous == None:
                previous = file_path[0]
            if previous != file_path[0]:
                all_folds.append(curr_fold)
                curr_fold = []
                previous = file_path[0]
            else:
                curr_fold.append([base_dir+file_path, label])
        all_folds.append(curr_fold) # the last fold
    return all_folds
    """
    for i in range(len(all_folds)):
        print('==================={0} fold================'.format(i+1))
        for j in range(len(all_folds[i])):
            print(all_folds[i][j])
    """


def ck_aug_image_path_and_label():
    base_dir = './CKPLUS_2015CCV/'
    label_file = 'label.txt'
    target_file = base_dir + label_file
    all_folds = [[] for i in range(10)]
    with open(target_file, 'r') as f:
        for line in f:
            file_name, label, group = line.strip().split()
            # skip some files that do not exist
            if file_name.split('/')[1].startswith(('n15', 'p15')):
                continue
            file_path = base_dir + file_name
            all_folds[int(group)-1].append([file_path, label])
    """
    for i in range(len(all_folds)):
        print('==================={0} fold================'.format(i+1))
        print(len(all_folds[i]))
        for j in range(len(all_folds[i])):
            file_path, label = all_folds[i][j]
            if not os.path.exists(file_path):
                print('file {0} not exists'.format(file_path))
    """
    return all_folds


def kdef_image_path_and_label():
    base_dir = './KDEF_G/'
    #base_dir = './CKPLUS_2015CCV_ORI/'
    label_file = 'label.txt'
    target_file = base_dir + label_file
    all_folds = [[] for i in range(10)]
    with open(target_file, 'r') as f:
        for line in f:
            file_name, label, group = line.strip().split()
            file_path = base_dir+'Group{0}/'.format(group)+file_name
            all_folds[int(group)-1].append([file_path, label])
    for i in range(len(all_folds)):
        print('==================={0} fold================'.format(i+1))
        for j in range(len(all_folds[i])):
            file_path, label = all_folds[i][j]
            if not os.path.exists(file_path):
                print('file {0} not exists'.format(file_path))
    return all_folds


def dump_image_pkl_file():
    image_size = (224, 224)
    path_label = ck_image_path_and_label()
    path_label = kdef_image_path_and_label()
    path_label = ck_aug_image_path_and_label()
    label_set = set()
    all_fold_feature = []
    for i in range(len(path_label)):
        print('==============={0} fold==============='.format(i+1))
        curr_fold_feature = []
        for j in range(len(path_label[i])):
            img_path, label = path_label[i][j]
            img = image.load_img(img_path, target_size = image_size)
            x = image.img_to_array(img)
            print(x.shape)
            # x = np.expand_dims(x, axis=0)
            curr_feature = {}
            curr_feature['image'] = x
            curr_feature['label'] = int(label)
            label_set.add(int(label))
            curr_fold_feature.append(curr_feature)
        all_fold_feature.append(np.array(curr_fold_feature))
    print(label_set)
    feature_pkl_file = './Datasets/ck+_ori_aug_10groups_224_224_3.pkl'
    with open(feature_pkl_file, 'wb') as wf:
        pickle.dump(np.array(all_fold_feature), wf)


def pickle_2_numpy(feature_pkl_file):
    with open(feature_pkl_file, 'rb') as rf:
        data = pickle.load(rf)
    X, Y = [], []
    for i in range(len(data)):
        #print(data[i].shape)
        curr_X, curr_Y = [], []
        for j in range(len(data[i])):
            curr_X.append(data[i][j]['image'])
            curr_Y.append(data[i][j]['label'])
        X.append(np.array(curr_X))
        Y.append(np.array(curr_Y))
    return (X, Y)
            


if __name__ == '__main__':
    # extract_face('ck.png', 'ck_face.png')
    # batch_face_extraction()
    # image_path_and_label()
    dump_image_pkl_file()
    # kdef_batch_face_extraction()
    # kdef_image_path_and_label()
    # ck_aug_image_path_and_label()
    """
    feature_pkl_file = './Datasets/ck_10groups_224_224_3.pkl'
    X, Y = pickle_2_numpy(feature_pkl_file)
    for i in range(len(Y)):
        print(X[i].shape, Y[i].shape)
    """