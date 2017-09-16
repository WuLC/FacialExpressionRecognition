# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-07-27 09:38:11
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-11 21:04:20

###############################################################################
# Extract feature of image with models pretrained on ImageNet provided by Keras
# Then classify images with thest features and your classifier
# Actually such method belongs to Transfer Learning
###############################################################################

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import pickle
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from ExtractFace import ck_image_path_and_label, kdef_image_path_and_label

def feature_extraction():
    image_size = (224, 224)
    # load existing model
    model = VGG16(weights='imagenet', include_top=False)

    # get 10-fold image path and corresponding label
    path_label = ck_image_path_and_label()
    path_label = kdef_image_path_and_label()
    feature_label = {'feature' : [], 'label' : []}
    all_folds = []
    for i in range(len(path_label)):
        print('==============={0} fold==============='.format(i+1))
        curr_fold = []
        curr_feature = []
        curr_label = []
        for j in range(len(path_label[i])):
            img_path, label = path_label[i][j]
            img = image.load_img(img_path, target_size = image_size)
            x = image.img_to_array(img)
            # print(type(x), x.shape)
            x = np.expand_dims(x, axis=0)
            # print(type(x), x.shape)
            x = preprocess_input(x)
            # print(type(x), x.shape)
            features = model.predict(x)
            # print(type(features), features[0].shape)
            # print(features.flatten().shape, np.insert(features.flatten(), 0, 0))
        """ #store feature as numpy array
            curr_fold.append(np.insert(features[0].flatten(), 0, int(label)))
        all_folds.append(np.array(curr_fold))
        """
            # store feature as dictionary
            curr_feature.append(features[0])
            curr_label.append(int(label))
        feature_label['feature'].append(np.array(curr_feature))
        feature_label['label'].append(np.array(curr_label))
    
    feature_pkl_file = './Datasets/KDEF_VGG16_224_224_face_7_7_512.pkl'
    with open(feature_pkl_file, 'wb') as wf:
        pickle.dump(feature_label, wf)
    

def load_dict_feature(data_file):
    with open(data_file, 'rb') as rf:
        data = pickle.load(rf)
    # print(type(data), type(data['feature']), type(data['label']))
    return data['feature'], data['label']


def load_numpy_fearure(data_file):
    if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
    with open(data_file, 'rb') as rf:
        data = pickle.load(rf)

    print(len(data[0][0]))
    total_x, total_y = [], []
    # k-flod cross validation
    for i in range(len(data)): # traverse the 10 splits
        curr_x, curr_y  = [], []
        for j in range(len(data[i])): # traverse sample of each split
            curr_x.append(data[i][j][1:])
            curr_y.append(int(data[i][j][0]))
        total_x.append(np.array(curr_x))
        total_y.append(np.array(curr_y))
    return total_x, total_y


if __name__ == '__main__':
    data_file = './Datasets/VGG16_224_224_face_7_7_512.pkl'
    feature_extraction()
    # load_dict_feature(data_file)
