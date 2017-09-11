# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-07-14 14:31:40
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-11 21:24:35


#########################################################################################
# Mainly compared with BatchLearning, OnlineLearning doesn't require all the data at once
# instead, it is suitable for streaming data
# the core algorithm is stochastic gradient descent
#########################################################################################

import time
import logging
import numpy as np

from sklearn import linear_model
from sklearn import metrics

from PreProcessing import pickle_2_numpy, pca_reduce, standardrize_input


class OnlineLinearModel:
    def __init__(self, data_file):
        self.x, self.y = pickle_2_numpy(data_file)


    def train(self, loss_metric, logfile = None, n_iter = 20, standard = False, pca = False, n_dimension = 100):
        """train the model and evaluate it with accuracy and confusion matrix
        
        Args:
            loss_metric (str): loss metrix of the SGD classifier: log, hinge, squared_hinge, modified_huber
            n_iter (int, optional): number of iteration
            standard (bool, optional): standardrize input
            pca (bool, optional): whether to perform pca on the input data
            n_dimension (int, optional): number of components that PCA want to achieve
        
        Returns:
            None
        """

        if logfile == None:
            print('specify the path of logfile firstly')
            exit(-1)
        logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")

        # dimension reduction with pca
        if pca:
            logging.info('start PCA')
            start_time = time.time()
            for i in range(len(self.y)):
                self.x[i] = pca_reduce(self.x[i], n_dimension = n_dimension)
            logging.info('end PCA, time consuming {:.2f}s'.format(time.time() - start_time))

        # standardrize the data to obtain zero mean and unit variance
        if standard:
            logging.info('start standard')
            start_time = time.time()
            for i in range(len(self.y)):
                self.x[i] = standardrize_input(self.x[i])
            logging.info('end standard, time consuming {:.2f}s'.format(time.time() - start_time))

        # perform cross validataion
        y_predict = []
        cross_val_score = []
        y_8_predict = None
        y_8_true = None
        start_time = time.time()
        for i in range(len(self.y)):
            model = linear_model.SGDClassifier(loss = loss_metric, penalty ='l1', n_jobs = -1)
            for _ in range(n_iter):
                # online learning, default iteration is 1
                for j in range(len(self.y)):
                    if i == j:
                        continue
                    model.partial_fit(self.x[j], self.y[j], classes = np.array(range(0,7)))
            predict = model.predict(self.x[i])
            cross_val_score.append(float('{:.3f}'.format(metrics.accuracy_score(predict, self.y[i]))))
            y_predict.append(predict)
            if i == (len(self.y) - 1):
                y_last_predict = predict
                y_last_true = self.y[i]
        y_predict = np.concatenate(y_predict)
        y_true = np.concatenate(self.y)
        logging.info('training model description \n {0}'.format(model))
        logging.info('number of iteration: {0}'.format(n_iter))
        logging.info('Time Consuming {0}s'.format(time.time() - start_time))
        logging.info('Accuracy \n 8-fold: {0} \n average {1}\n'.format(cross_val_score, np.mean(cross_val_score)))
        cm = metrics.confusion_matrix(y_true, y_predict)
        cm_ratio = (cm/cm.sum(axis = 1, keepdims = True))*100
        logging.info('Confusion Matrix \n {0}\n {1}\n'.format(cm, cm_ratio))
        cm = metrics.confusion_matrix(y_last_true, y_last_predict)
        cm_ratio = (cm/cm.sum(axis = 1, keepdims = True))*100
        logging.info('Confusion Matrix for the last fold \n {0}\n {1}'.format(cm, cm_ratio))
        return np.mean(cross_val_score)