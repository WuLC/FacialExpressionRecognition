# -*- coding: utf-8 -*-
# Created on Sun Jun 03 2018 21:47:52
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import time
import logging

import lightgbm as lgb
import numpy as np
from sklearn import metrics

from PsychologyFeature import genrate_data

class LGBClassifier():
    def __init__(self):
        self.num_boost_round = 200
        self.early_stopping_rounds = 15
        self.params = {
            'task': 'train',
            'boosting_type': 'dart',
            'objective': 'multiclass',
            'num_class': 7,
            'metric': {'multi_logloss', 'multi_error'},
            # 'num_leaves': 80,
            # 'learning_rate': 0.05,
            # 'scale_pos_weight': 1.5,
            # 'feature_fraction': 0.5,
            # 'bagging_fraction': 1,
            # 'bagging_freq': 5,
            # 'max_bin': 300,
            # 'is_unbalance': True,
            # 'lambda_l2': 5.0,
            # 'verbose' : -1
            }
        
    def fit(self, x_train, y_train, x_val, y_val):
        print('train with lgb model')
        lgbtrain = lgb.Dataset(x_train, y_train)
        lgbval = lgb.Dataset(x_val, y_val)
        self.model = lgb.train(self.params, 
                          lgbtrain,
                          valid_sets = lgbval,
                          verbose_eval = self.num_boost_round,
                          num_boost_round = self.num_boost_round,
                          early_stopping_rounds = self.early_stopping_rounds)
    
    def predict(self, x_test):
        return self.model.predict(x_test, num_iteration=self.model.best_iteration)


def cross_validation(X, Y, logfile = None):
    if logfile == None:
        print('specify the path of logfile firstly')
        return
    logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")

    start_time = time.time()
    y_predict = []
    cross_val_score = []
    for i in range(len(Y)):
        model = LGBClassifier()
        train_X, train_Y= [], []
        test_X, test_Y = None, None
        for j in range(len(Y)):
            if i == j:
                test_X, test_Y = np.array(X[j]), np.array(Y[j])
            else:
                train_X.append(X[j])
                train_Y.append(Y[j])
        train_X = np.concatenate(train_X, axis = 0)
        train_Y = np.concatenate(train_Y)
        model.fit(train_X, train_Y, test_X, test_Y)
        prediction = model.predict(test_X)
        y_predict.append(prediction)
        cross_val_score.append(float('{:.3f}'.format(metrics.accuracy_score(prediction, test_Y))))
    y_predict = np.concatenate(y_predict)
    y_true = np.concatenate(Y)
    logging.info('training model description \n {0}'.format(model))
    logging.info('length of feature {0}'.format(len(X[0][0])))
    logging.info('time consuming: {0}'.format(time.time() - start_time))
    logging.info('Accuracy \n 10-fold: {0} \n average {1}'.format(cross_val_score, np.mean(cross_val_score)))
    logging.info('Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(y_true, y_predict)))


def main():
    X, Y = genrate_data()
    cross_validation(X, Y, logfile='../logs/PsychologyFeature+GBDT.log')

if __name__ == '__main__':
    main()