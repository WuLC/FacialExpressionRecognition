# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-07-25 17:00:26
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-11 21:12:42

############################################################################################
# train AlexNet from scratch for image classification, that is facial expression recognition
############################################################################################

import time
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='1' # use the second GPU

import numpy as np
import logging
import pickle
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.models import Model
from keras.optimizers import SGD

from PreProcessing import pickle_2_numpy

train_data_file =  './Datasets/D_KDEF_10G_only_rescale_images_with_RBG.pkl'
test_data_file =  './Datasets/D_KDEF_10G_only_rescale_images_with_RBG.pkl'
model_base_path = './models/{0}_{1}epoches_'.format(train_data_file.split('/')[-1].split['.'][0], epochs)
logfile = './logs/{0}.log'.format(train_data_file.split('/')[-1].split['.'][0])
logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")

height, width, channels = 128, 128, 3
categories = 7
batch_size = 32
epochs = 50
feature_dim = 1000
feature_file_dir = './Datasets/dl_feature/'
feature_file_name = 'CK+_KDEF_JAFFFE_alexnet_feature_dim_{0}.pkl'.format(feature_dim)

class AlexNet:
    def __init__(self):
        # load and reshape data firstly
        x, y  = pickle_2_numpy(train_data_file , original_image = True)
        test_x, test_y = pickle_2_numpy(test_data_file , original_image = True)

        self.X, self.Y = [], []
        self.test_X, self.test_Y = [], []
        self.groups = len(y)
        for i in range(self.groups):
            # transform train data
            self.X.append(x[i].reshape(x[i].shape[0], height, width, channels))
            self.X[i] = self.X[i].astype('float32')
            self.X[i] /= 255
            self.Y.append(np_utils.to_categorical(y[i], categories))
            # print (X[i].shape, Y[i].shape)

            # transform test data
            self.test_X.append(test_x[i].reshape(test_x[i].shape[0], height, width, 1))
            self.test_X[i] = self.test_X[i].astype('float32')
            self.test_X[i] /= 255
            self.test_Y.append(np_utils.to_categorical(test_y[i], categories))

    def build_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(48, (11, 11), activation='relu', input_shape = (height, width, channels)))
        self.model.add(BatchNormalization(axis = -1))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Convolution2D(128, (5, 5), activation='relu'))
        self.model.add(BatchNormalization(axis = -1))
        self.model.add(MaxPooling2D(pool_size = (2,2)))
        self.model.add(Convolution2D(192, (3, 3), activation='relu'))
        self.model.add(Convolution2D(192, (3, 3), activation='relu'))
        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size = (2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(feature_dim, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(feature_dim, activation='relu', name = 'feature'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(categories, activation='softmax'))

    def train_and_eval(self, dump_feature = False):
        self.model.compile(loss='categorical_crossentropy',
                    optimizer = SGD(lr=0.0001, momentum=0.9),
                    metrics=['accuracy'])

        initial_weights = self.model.get_weights()
        # cross validation
        cv_score = []
        extracted_features = []
        start_time = time.time()
        for i in range(self.groups):
            print('=============={0} fold=============='.format(i+1))
            X_test = self.test_X[i]
            Y_test = self.test_Y[i]
            # check whether the model already exists
            model_path = model_base_path + '{0}.h5'.format(i)
            if os.path.exists(model_path):
                self.model.load_weights(model_path)
            else:
                self.model.set_weights(initial_weights)
                X_train, Y_train = [], []
                for j in range(self.groups):
                    if i == j:
                        continue
                    X_train.append(self.X[j])
                    Y_train.append(self.Y[j])
                X_train = np.concatenate(self.X_train, axis = 0)
                Y_train = np.concatenate(self.Y_train, axis = 0)
                #print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

                self.model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 2, validation_data = (X_test, Y_test))
                self.model.save_weights(model_path)
            
            # validation set can be very large, need batch size
            score = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose=0)
            cv_score.append(score[1])
            print('**************validation accuracy of fold {0}:{1}******************'.format(i+1, score[1]))
            print('**************curr average accuracy {0}******************'.format(np.mean(cv_score)))

            if dump_feature:
                # get feature output
                layer_name = 'feature'
                feature_layer_model = Model(inputs = model.input, outputs = self.model.get_layer(layer_name).output)
                feature_output = feature_layer_model.predict(X_test)
                label_feature = []
                for j in range(self.groups):
                    label_feature.append(np.insert(feature_output[j], 0, np.where(Y_test[j] == 1)[0][0]))
                extracted_features.append(label_feature)

                if len(extracted_features) == self.groups:
                    # dump feature extracted with the model
                    try:
                        if not os.path.exists(feature_file_dir):
                            os.makedirs(feature_file_dir)
                        with open(feature_file_dir+feature_file_name, 'wb') as wf:
                            pickle.dump(np.array(extracted_features), wf)
                        logging.info('successfully dumping the feature file')
                    except Exception:
                        logging.error('fail to dump the feature file')
        
        logging.info('model summary \n {0}'.format(self.model.summary()))
        logging.info('time consuming:{0}s'.format(time.time() - start_time))
        logging.info('k-fold accuracy:{0}'.format(cv_score))
        logging.info('average accuracy: {0}'.format(np.mean(cv_score)))
        logging.info('######################################################################\n')

