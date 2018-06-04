# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-07-18 10:09:25
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-11 21:21:14

#############################################################
# use some classical models for classification with sklearn
# with one-verse-rest for multiple classification
# features of humac face is stored in the pickle file
#############################################################

import time
import logging
import numpy as np

from sklearn import linear_model
from sklearn import svm 
from sklearn import neighbors
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn import metrics

from PreProcessing import pickle_2_numpy, pca_reduce, standardrize_input, load_numpy, discretize_input

class BatchLearning:
    def __init__(self, data_file, np_feature = False, only_geometry = False, discrete = False, standardrize = True):
        if np_feature:
            print('==load np feature==')
            self.X, self.Y = load_numpy(data_file)
        else:
            self.X, self.Y = pickle_2_numpy(data_file, only_geometry = only_geometry)

        if discrete:
            print('======perform discretize========')
            for i in range(len(self.Y)):
                self.X[i] = discretize_input(self.X[i])

        if standardrize:
            print('======perform standarize========')
            for i in range(len(self.Y)):
                self.X[i] = standardrize_input(self.X[i])

        print('length of feature {0}'.format(len(self.X[0][0])))

    def train(self, generate_model = None, logfile = None, pca = False, rand_project = False):
        if logfile == None:
            print('specify the path of logfile firstly')
            return
        logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")

        if generate_model == None:
            print('specify the model firstly')
            return

        # random projection 
        if rand_project:
            for i in range(len(self.X)):
                self.X[i] = random_project(self.X[i])

        start_time = time.time()
        y_predict = []
        cross_val_score = []
        for i in range(len(self.Y)):
            model = generate_model()
            train_X, train_Y= [], []
            test_X, test_Y = None, None
            for j in range(len(self.Y)):
                if i == j:
                    test_X, test_Y = self.X[j], self.Y[j]
                else:
                    train_X.append(self.X[j])
                    train_Y.append(self.Y[j])
            train_X = np.concatenate(train_X, axis = 0)
            train_Y = np.concatenate(train_Y)
            # pca reduction
            if pca:
                train_X = pca_reduce(train_X, n_dimension = len(train_X)-1)
            model.fit(train_X, train_Y)
            predict = model.predict(test_X)
            y_predict.append(predict)
            cross_val_score.append(float('{:.3f}'.format(metrics.accuracy_score(predict, test_Y))))
        y_predict = np.concatenate(y_predict)
        y_true = np.concatenate(self.Y)
        logging.info('training model description \n {0}'.format(model))
        logging.info('length of feature {0}'.format(len(self.X[0][0])))
        logging.info('time consuming: {0}'.format(time.time() - start_time))
        logging.info('Accuracy \n 10-fold: {0} \n average {1}'.format(cross_val_score, np.mean(cross_val_score)))
        logging.info('Confusion Matrix \n {0}\n'.format(metrics.confusion_matrix(y_true, y_predict)))



def logistic_regression():
    return linear_model.LogisticRegression(penalty = 'l1', n_jobs = -1)   


def svm_model():
    # return svm.SVC()
    return svm.LinearSVC()


def knn_model():
    return neighbors.KNeighborsClassifier(n_neighbors = 10)


def decision_tree():
    return tree.DecisionTreeClassifier()


def bagging_classifier():
    #base_model = linear_model.LogisticRegression(penalty = 'l1', n_jobs = -1)
    base_model = svm.LinearSVC()
    return BaggingClassifier(base_model, max_samples=0.5, max_features=0.5)


def boosting_classfier():
    base_model = linear_model.LogisticRegression(penalty = 'l1', n_jobs = -1)
    return AdaBoostClassifier(base_estimator = base_model)


def neural_network():
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 50))


if __name__ == '__main__':
    # data_file = './Datasets/D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'
    # data_file = './Datasets/D16_CKPLUS_10G_Enlargeby2015CCV_10T.pkl'
    # data_file = './Datasets/D30_KDEF_10groups_groupedbythe_KDEF-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface_innerface_skip-contempV.pkl'
    # data_file = './Datasets/D34_KDEF_10G_Enlargeby2015CCV_10T.pkl'
    # data_file = './Datasets/D43_JAFFE_10G_Enlargeby2015CCV_10T.pkl'
    # data_file = './Datasets/D40_JAFFE_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl'
    # data_file = './Datasets/D10_new_feature_CKplus_10groups.pkl'
    # data_file = './Datasets/VGGFeature/ckplus8G_label+vggfc4096+geometryfc1024.pkl'
    # data_file = './Datasets/dl_feature/dl_feature_dim_200.pkl'
    # data_file = './Datasets/dl_feature/dl_feature_dim_500.pkl'
    # data_file = './Datasets/dl_feature/dl_feature_dim_1000.pkl'
    # data_file = './Datasets/dl_feature/alexnet_feature_dim_2048.pkl'
    # data_file = './Datasets/dl_feature/VGG19_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/VGG19_128_128_face.pkl'
    # data_file = './Datasets/dl_feature/VGG16_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/VGG16_128_128_face.pkl'
    # data_file = './Datasets/dl_feature/InceptionV3_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/InceptionV3_128_128_face.pkl'
    # data_file = './Datasets/dl_feature/CK+_alexnet_feature_dim_1000.pkl'
    # data_file = './Datasets/dl_feature/CK+_dataaug_alexnet_feature_dim_1000.pkl'
    # data_file = './Datasets/dl_feature/KDEF_VGG16_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/KDEF_InceptionV3_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/KDEF_VGG19_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/KDEF_Xception_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/KDEF_MobileNet_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/CK+_Xception_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/CK+_MobileNet_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/CK+_ResNet50_224_224_face.pkl'
    # data_file = './Datasets/dl_feature/CK+_dataaug_alexnet_feature_dim_1000.pkl'
    # data_file = './Datasets/dl_feature/KDEF_dataaug_alexnet_feature_dim_1000.pkl'
    # data_file = './Datasets/dl_feature/JAFFE_dataaug_alexnet_feature_dim_1000.pkl'
    # data_file = './Datasets/D18_CKplus_10G_V5_formalized_weberface128x128.pkl'
    # data_file = './Datasets/D5_CKplus_8groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_webberface25up_skip-contempV2.pkl'
    # data_file = './Datasets/D13_CKplus_8G_V4_Geo258_ELTFS128x128.pkl'
    # data_file = './Datasets/D130_CKplus_8G_V4_Geo122withformalized_ELTFS128x128.pkl'
    # data_file = './Datasets/D131_CKplus_8G_V4_Geo122withstdXYformalized_ELTFS128x128.pkl'
    # data_file = './Datasets/D132_CKplus_8G_V4_Geo122withallstdXYformalized_withoutPatches.pkl'
    # data_file = './Datasets/D150_CKplus_10G_V5_geo262_weberface128x128.pkl'
    data_file = './Datasets/D111_MergeDataset_D10_D33_D40_10G.pkl'
    data_file = './Datasets/D222_MergeDataset_D16_D34_D43_10G.pkl'

    batch_model = BatchLearning(data_file, np_feature = False, only_geometry = False, discrete = False, standardrize = True)
    # logistic regression
    logfile = './logs/{0}_LogisticRegression.log'.format(data_file.split('/')[-1].split('.')[0])
    batch_model.train(generate_model = logistic_regression, logfile = logfile)
    
    
    # svm
    # logfile = './logs/D10_{0}_svm.log'.format(data_file.split('/')[-1].split('.')[0])
    # batch_model.train(generate_model = svm_model, logfile = logfile)
    """
    # knn
    logfile = 'D20_knn.log'
    batch_model.train(generate_model = knn_model, logfile = logfile)
    
    # decision tree
    logfile = 'D20_dt.log'
    batch_model.train(generate_model = decision_tree, logfile = logfile)

    # bagging
    logfile = 'D20_bagging.log'
    batch_model.train(generate_model = bagging_classifier, logfile = logfile)
    
    # boosting
    logfile = 'D10_boosting.log'
    batch_model.train(generate_model = boosting_classfier, logfile = logfile)
    
    # neural network
    logfile = 'D20_neuralnetwork.log'
    batch_model.train(generate_model = neural_network, logfile = logfile)
    """