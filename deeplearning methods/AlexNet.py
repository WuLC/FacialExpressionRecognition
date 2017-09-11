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

from PreProcessing import pickle_2_numpy


# CK+
data_file = './Datasets/D16_CKPLUS_10G_Enlargeby2015CCV_10T.pkl'
data_file = './Datasets/D18_CKplus_10G_V5_formalized_weberface128x128.pkl'
test_data_file = './Datasets/D10_CKplus_10groups_groupedbythe_CKplus-group-details_preprocessdata_with_calibRotation_rescaleimg_geometricfeatures_facepatches_weberface_skip-contempV2.pkl'

# KDEF
data_file = './Datasets/D34_KDEF_10G_Enlargeby2015CCV_10T.pknpl'
# data_file = './Datasets/D33_KDEF_10G_rescaleimg_geometryfeature_patches_web.pkl'
test_data_file = './Datasets/D33_KDEF_10G_rescaleimg_geometryfeature_patches_web.pkl'


# JAFFE
data_file = './Datasets/D43_JAFFE_10G_Enlargeby2015CCV_10T.pkl'
# data_file = './Datasets/D40_JAFFE_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl'
test_data_file = './Datasets/D40_JAFFE_10groups_groupedbysubjects_rescaleimg_geometricfeatures_facepatches_weber.pkl'

# merged file
data_file =  './Datasets/D111_MergeDataset_D10_D33_D40_10G.pkl'
test_data_file =  './Datasets/D111_MergeDataset_D10_D33_D40_10G.pkl'

data_file =  './Datasets/D222_MergeDataset_D16_D34_D43_10G.pkl'
test_data_file =  './Datasets/D222_MergeDataset_D16_D34_D43_10G.pkl'

height, width = 128, 128
categories = 7
batch_size = 32
epochs = 50
feature_dim = 1000
feature_file_dir = './Datasets/dl_feature/'
feature_file_name = 'CK+_KDEF_JAFFFE_alexnet_feature_dim_{0}.pkl'.format(feature_dim)
model_base_path = './models/CK+_KDEF_JAFFFE_alexnet_{0}epochs_'.format(epochs)
logfile = './logs/CK+_KDEF_JAFFFE_alexnet.log'
logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")

# load and reshape data firstly
x, y  = pickle_2_numpy(data_file , original_image = True)
test_x, test_y = pickle_2_numpy(test_data_file , original_image = True)

X, Y = [], []
test_X, test_Y = [], []
for i in range(len(y)):
    # transform train data
    X.append(x[i].reshape(x[i].shape[0], height, width, 1))
    X[i] = X[i].astype('float32')
    X[i] /= 255
    Y.append(np_utils.to_categorical(y[i], categories))
    # print (X[i].shape, Y[i].shape)

    # transform test data
    test_X.append(test_x[i].reshape(test_x[i].shape[0], height, width, 1))
    test_X[i] = test_X[i].astype('float32')
    test_X[i] /= 255
    test_Y.append(np_utils.to_categorical(test_y[i], categories))
    
# generate model
model = Sequential()

# alexnet
model.add(Convolution2D(48, (11, 11), activation='relu', input_shape = (height, width, 1)))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(128, (5, 5), activation='relu'))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(192, (3, 3), activation='relu'))
model.add(Convolution2D(192, (3, 3), activation='relu'))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(feature_dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(feature_dim, activation='relu', name = 'feature'))
model.add(Dropout(0.5))
model.add(Dense(categories, activation='softmax'))



# specify optimizer
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy',
              optimizer = SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])

initial_weights = model.get_weights()

# cross validation
cv_score = []
extracted_features = []
start_time = time.time()
for i in range(len(Y)):
    print('=============={0} fold=============='.format(i+1))
    X_test = X[i]
    Y_test = Y[i]
    # check whether the model already exists
    model_path = model_base_path + '{0}.h5'.format(i)
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        model.set_weights(initial_weights)
        X_train, Y_train = [], []
        for j in range(len(Y)):
            if i == j:
                continue
            X_train.append(X[j])
            Y_train.append(Y[j])
        X_train = np.concatenate(X_train, axis = 0)
        Y_train = np.concatenate(Y_train, axis = 0)
        #print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

        model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 2, validation_data = (X_test, Y_test))
        model.save_weights(model_path)

    # get feature output
    layer_name = 'feature'
    feature_layer_model = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)
    feature_output = feature_layer_model.predict(X_test)
    label_feature = []
    for j in range(len(Y_test)):
        label_feature.append(np.insert(feature_output[j], 0, np.where(Y_test[j] == 1)[0][0]))
    extracted_features.append(label_feature)
    
    # validation set can be very large, need batch size
    score = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose=0)
    cv_score.append(score[1])
    print('**************validation accuracy of fold {0}:{1}******************'.format(i+1, score[1]))
    print('**************curr average accuracy {0}******************'.format(np.mean(cv_score)))

# dump feature extracted with the model

try:
    if not os.path.exists(feature_file_dir):
        os.makedirs(feature_file_dir)
    with open(feature_file_dir+feature_file_name, 'wb') as wf:
        pickle.dump(np.array(extracted_features), wf)
    logging.info('successfully dumping the feature file')
except Exception:
    logging.error('fail to dump the feature file')

logging.info('model summary \n {0}'.format(model.summary()))
logging.info('time consuming:{0}s'.format(time.time() - start_time))
logging.info('k-fold accuracy:{0}'.format(cv_score))
logging.info('average accuracy: {0}'.format(np.mean(cv_score)))
logging.info('######################################################################\n')