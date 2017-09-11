# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-07-10 08:29:04
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-11 21:16:02


import os
import pickle 
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import random_projection

def explore_data(data_file):
    # explore structure of the pkl data file
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    count = defaultdict(int)
    all_labels = set()
    total_sample_count = 0
    for i in range(len(data)):
        total_sample_count += len(data[i]['labels'])
        print ('Group {0}'.format(i+1))
        curr = defaultdict(int)
        print (len(data[i]['imgs']), len(data[i]['geometry']), len(data[i]['eye_patch']),\
               len(data[i]['middle_patch']), len(data[i]['mouth_patch']), len(data[i]['inner_face']),\
               len(data[i]['labels']))
        for j in range(len(data[i]['labels'])):
            count[data[i]['labels'][j]] += 1
            curr[data[i]['labels'][j]] += 1
        print ('label count {0}\n'.format(sorted(curr.items())))
        all_labels |= set(data[i]['labels'])
    print('total sample count:{0} \ntotal label count: {1}'.format(total_sample_count, sorted(count.items())))

    print (data[0]['imgs'][0].shape)
    print (len(data[0]['geometry'][0]))
    print (data[0]['eye_patch'][0].shape)
    print (data[0]['middle_patch'][0].shape)
    print (data[0]['mouth_patch'][0].shape)
    print (data[0]['inner_face'][0].shape)
    print (data[0]['labels'][0])
    print (all_labels)


def pickle_2_numpy(data_file, original_image = False, only_geometry = False):
    """transfer pkl file into numpy array
    
    Args:
        data_file (str):  path of the pkl data file
    
    Returns:
        x(numpy array), y(numpy array): training data and their labels
    """
    if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    total_x, total_y = [], []
    # 8-flod cross validation
    for i in range(len(data)): # traverse the 8 splits
        curr_x  = []
        for j in range(len(data[i]['labels'])): # traverse sample of each split
            if original_image:
                curr_x.append(np.array(data[i]['imgs'][j]))
            elif only_geometry:
                curr_x.append(np.array(data[i]['geometry'][j]))
            else:
                curr_x.append(np.concatenate([data[i]['imgs'][j].flatten(), data[i]['geometry'][j],\
                          data[i]['eye_patch'][j].flatten(), data[i]['middle_patch'][j].flatten(), data[i]['mouth_patch'][j].flatten()]))
        total_x.append(np.array(curr_x))
        total_y.append(np.array(list(map(int, data[i]['labels']))))
    """
    for i in range(len(data)):
        print (np.array(total_x[i]).shape, np.array(total_y[i]).shape)
    """
    return total_x, total_y


def load_numpy(data_file):
    if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
    with open(data_file, 'rb') as rf:
        data = pickle.load(rf)

    #print('length of feature {0}'.format(len(data[0][0])))
    total_x, total_y = [], []
    # k-flod cross validation
    for i in range(len(data)): # traverse the 10 splits
        curr_x, curr_y  = [], []
        for j in range(len(data[i])): # traverse sample of each split
            curr_x.append(data[i][j][1:])
            curr_y.append(int(data[i][j][0]))
        total_x.append(np.array(curr_x))
        total_y.append(np.array(curr_y))
    return np.array(total_x), np.array(total_y)


def standardrize_input(x):
    # print('before scaling \n mean:{0} \n variance:{1}'.format(x.mean(axis = 0), x.std(axis = 0)))
    # x = preprocessing.normalize(x)
    x = preprocessing.scale(x)
    # print('after scaling \n mean:{0} \n variance:{1}'.format(x.mean(axis = 0), x.std(axis = 0)))
    return x


def discretize_input(x):
    max_value = 0
    discrete_x = np.zeros((x.shape[0], x.shape[1] * 10))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            max_value = max(max_value, x[i][j])
            offset = int(x[i][j] / 25)
            if offset == 10: 
                offset -= 1
            idx = j * 10 + offset
            discrete_x[i][idx] = 1
    print('max value {0}'.format(max_value))
    return discrete_x


def pca_reduce(x, n_dimension):
    pca = PCA(n_components = n_dimension, whiten = True)
    return pca.fit_transform(x)


def random_project(x):
    print('original shape:{0}'.format(x.shape))
    transformer = random_projection.GaussianRandomProjection()
    x_new = transformer.fit_transform(x)
    print('after random projection:{0}'.format(x_new.shape))
    return x_new


def test_label(label_file):
    mapping = {}
    with open(label_file, 'r') as f:
        for line in f:
            if line.startswith('G8'):
                name, label = line.split()
                mapping[name[3:-4]] = int(label)
    #print(mapping)
    for k,v in mapping.items():
        d1, d2, d3 = k.split('_')
        file = 'D:/FacicalExpressionRecognition/dataset/CK+/Emotion/{0}/{1}/{2}_emotion.txt'.format(d1, d2, k)
        if not os.path.exists(file):
            print('image:{1}, file {0} not exists'.format(file, k))
            continue
        with open(file, 'r') as f:
            label = int(f.readline().strip()[0])
            if label != v:
                print ('image {0} label not match'.format(k))


if __name__ == '__main__':
    # number of instance 1236
    # number of feature 28410
    data_file = './Datasets/D30_KDEF_10groups_groupedbythe_KDEF-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerface_skip-contempV.pkl'
    # data_file = './Datasets/D20_MMI_8groups_groupedbythe_MMI-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerface_skip-contempV.pkl'
    # data_file = './Datasets/D5_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface25up_skip-contempV2.pkl'
    # data_file = './Datasets/D10_CKplus_10groups.pkl'
    # data_file = './Datasets/D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'
    # data_file = './D4_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberfaceReverse_skip-contempV2.pkl'
    # explore_data(data_file)
    #label_file = 'D:/FacicalExpressionRecognition/dataset/CK+/8groups/label.txt'
    #test_label(label_file)
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    with open(new_data_file, 'wb') as f:
        pickle.dump(data, f, protocol = 2)
    """
    x, y = pickle_2_numpy(data_file, original_image = True)
    
    new = []
    # x, y = load_numpy(data_file)
    for i in range(len(x)):
        print(type(x[i]), x[i].shape, type(y[i]), len(y[i]))
        new.append(x[i].reshape(98, 128, 128, 1))
        # print(x[i][0])
    