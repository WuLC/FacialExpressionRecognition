# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-08 20:13:33
# @Last Modified by:   lc
# @Last Modified time: 2017-09-15 12:16:30

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from datetime import datetime

import cv2
import numpy as np 
from keras.models import model_from_json

MAPPING ={0:'neural', 1:'anger', 2:'surprise', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness'}

class AlexNet:
    def __init__(self,
                 json_path = './models/D222_CK+_KDEF_JAFFFE_alexnet_200epochs_json',
                 weight_path = './models/D222_CK+_KDEF_JAFFFE_alexnet_200epochs_weight'):
        # load model
        self.model = None
        with open(json_path, 'r') as f:
            self.model = model_from_json(f.read())
            print('load model architecture from {0} successfully!!!!!!!!!!!!!!'.format(json_path))
        if self.model:
            self.model.load_weights(weight_path)
            print('load model weights from {0} successfully!!!!!!!!!!!!!!!!!!!!'.format(weight_path))


    def predict(self, img):
        """
        
        Args:
            img (ndarray): 
        
        Returns:
            TYPE
        """
        probability = self.model.predict(img) # input need to be a tuple whose length is 4
        print('probability distribution:{0} \n sum of probability: {1}'.format(probability, np.sum(probability)))
        cate = np.argmax(probability)
        # print('category: {0}'.format(cate))
        return MAPPING[cate], probability


if __name__ == '__main__':
    
    json_path = './models/D10_CKplus_alexnet_150epochs_json'
    weight_path = './models/D10_CKplus_alexnet_150epochs_weight'
    model = AlexNet(json_path, weight_path)
    
    #model = AlexNet()
    src_dir = './src_img/'
    for file in os.listdir(src_dir):
        with open(src_dir + file, 'rb') as f:
            img = f.read()
            np_arr = np.fromstring(img, np.uint8) # one dimension array
            np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            emotion, _, _ = model.predict(np_img)
            print('category for img {0} is {1}'.format(file, emotion))
