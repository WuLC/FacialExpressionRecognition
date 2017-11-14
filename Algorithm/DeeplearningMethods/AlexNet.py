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

HEIGHT, WIDTH, CHANNELS = 128, 128, 1
CATEGORIES = 7
BATCH_SIZE = 32
EPOCHS = 300
FEATURE_DIM = 1000

class AlexNet:
    def __init__(self, train_data_file, test_data_file, feature_file_dir, model_dir, log_dir):
        prefix_name = train_data_file.split('/')[-1].split('.')[0]
        logfile = '{0}{1}.log'.format(log_dir, prefix_name)
        logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")
        self.feature_file_base_path = '{0}{1}_AlexNet_dim_{2}'.format(feature_file_dir, prefix_name, FEATURE_DIM)
        self.model_base_path = '{0}{1}_{2}epoches_'.format(model_dir, prefix_name, EPOCHS)
        self.model_architecture = '{0}architecture.json'.format(model_dir)

        # load and reshape data firstly
        x, y  = pickle_2_numpy(train_data_file , original_image = True)
        test_x, test_y = pickle_2_numpy(test_data_file , original_image = True)

        self.X, self.Y = [], []
        self.test_X, self.test_Y = [], []
        self.groups = len(y)
        for i in range(self.groups):
            # transform train data
            self.X.append(x[i].reshape(x[i].shape[0], HEIGHT, WIDTH, CHANNELS))
            self.X[i] = self.X[i].astype('float32')
            self.X[i] /= 255
            self.Y.append(np_utils.to_categorical(y[i], CATEGORIES))
            # print (X[i].shape, Y[i].shape)

            # transform test data
            self.test_X.append(test_x[i].reshape(test_x[i].shape[0], HEIGHT, WIDTH, CHANNELS))
            self.test_X[i] = self.test_X[i].astype('float32')
            self.test_X[i] /= 255
            self.test_Y.append(np_utils.to_categorical(test_y[i], CATEGORIES))

    def build_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(48, (11, 11), activation='relu', input_shape = (HEIGHT, WIDTH, CHANNELS)))
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
        self.model.add(Dense(FEATURE_DIM, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(FEATURE_DIM, activation='relu', name = 'feature'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(CATEGORIES, activation='softmax'))

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
            model_path = self.model_base_path + '{0}.h5'.format(i)
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
                X_train = np.concatenate(X_train, axis = 0)
                Y_train = np.concatenate(Y_train, axis = 0)
                #print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

                self.model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 2, validation_data = (X_test, Y_test))
                # save model architecture and weights
                if not os.path.exists(self.model_architecture):
                    with open(self.model_architecture, encoding='utf8', mode = 'w') as wf:
                        wf.write(self.model.to_json())
                self.model.save_weights(model_path)
            
            # validation set can be very large, need batch size
            score = self.model.evaluate(X_test, Y_test, batch_size = BATCH_SIZE, verbose=0)
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
                        if not os.path.exists(self.feature_file_dir):
                            os.makedirs(self.feature_file_dir)
                        feature_file_name = '{0}_{1}_fold.pkl'.format(self.feature_file_base_path, i)
                        with open(feature_file_name, 'wb') as wf:
                            pickle.dump(np.array(extracted_features), wf)
                        logging.info('successfully dumping the feature file')
                    except Exception:
                        logging.error('fail to dump the feature file')
        
        logging.info('model summary \n {0}'.format(self.model.summary()))
        logging.info('time consuming:{0}s'.format(time.time() - start_time))
        logging.info('k-fold accuracy:{0}'.format(cv_score))
        logging.info('average accuracy: {0}'.format(np.mean(cv_score)))
        logging.info('######################################################################\n')