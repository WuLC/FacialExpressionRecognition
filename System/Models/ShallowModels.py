# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-26 10:20:24
# @Last Modified by:   lc
# @Last Modified time: 2017-09-26 14:23:19

import os
import random 

import numpy as np
from sklearn.externals import joblib

MAPPING = {0:'neutral', 1:'angry', 2:'surprise', 3:'disgust', 4:'fear', 5:'happy', 6:'sad'}


class LogisticRegression():
    def __init__(self, model_path = './Models/model_weights/DF2_Logistic_Regression.pkl'):
        if not os.path.exists(model_path):
            print('model file {0} do not exist'.format(model_path))
            exit()
        self.model = joblib.load(model_path)


    def predict(self, geometric_feature):
        """
        
        Args:
            geometric_feature (ndarray):  ndarray of shape (None, 1, 310)
        
        Returns:
            emotion and probability distribution
        """
        overall_probability = self.model.predict_proba(geometric_feature)
        # print('probability distribution:{0} \n sum of probability: {1}'.format(probability, np.sum(probability)))
        cate = [np.argmax(probability) for probability in overall_probability]
        return [MAPPING[c] for c in cate], overall_probability

if __name__ == '__main__':
    model = LogisticRegression()
    data = [[random.randint(1, 1000) for j in range(310)] for i in range(5)]
    print(model.predict(data))

