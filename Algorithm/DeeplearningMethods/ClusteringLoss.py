import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import pickle
import time
import numpy as np
import logging

from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.resnet50 import ResNet50
# from keras.applications.xception import Xception
# from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import np_utils
from keras.layers import Embedding, Input, Flatten, Dense, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD

from ClusteringLossCallback import Evaluation

# specify image size, categories, and log file 
height, width = 224, 224
feature_dim = 256
categories = 7
batch_size = 16
epochs = 20
model_base_path = '../models/CK+_VGG16_fine_tunning_epoch({0})_'.format(epochs)
top_model_base_path =  '../models/CK+_bottleneck_fc_model_'
logfile = '../logs/CK+_ClusteringLoss.log'
logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")


######################### cross validation ##################################

# load and reshape data firstly
feature_pkl_file = '../Datasets/CK+/pkl/unified_10g.pkl'
feature_pkl_file = '../Datasets/CK+/pkl/original_10g.pkl'
with open(feature_pkl_file, 'rb') as rf:
    X, Y = pickle.load(rf)
    Y_value = Y[:]
for i in range(len(Y)):
    Y[i] = np_utils.to_categorical(Y[i], categories)

use_clustering_loss = True
call_back_evaluation = Evaluation(use_clustering_loss)

# load pretrained model
initial_model = VGG16(weights='imagenet', include_top=False)
print('Model loaded.')
for layer in initial_model.layers[:-5]: # keep some layers non-trainable (weights will not be updated)
    layer.trainable = False

input = Input(shape=(height, width, 3),name = 'image_input')
tmp = initial_model(input)
tmp = Flatten()(tmp)
tmp = Dense(feature_dim, activation='relu', name = 'feature')(tmp)
# tmp = Dropout(0.5)(tmp)
predictions = Dense(categories, activation = 'softmax')(tmp)

if not use_clustering_loss:
    model = Model(input, predictions)
    model.compile(loss='categorical_crossentropy',
            optimizer = SGD(lr=0.0001, momentum=0.9),
            metrics=['accuracy'])
else:
    print('=====use clustering loss=====')
    lambda_inner, lambda_outer = 0.002, -0.000002
    Centers = Embedding(categories, feature_dim)

    # inner loss
    input_target = Input(shape=(1,)) # single value ground truth labels as inputs
    center = Centers(input_target)
    inner_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True),name='inner_loss')([tmp, center])
    # print('*******************', center.shape, ip1.shape, inner_loss.shape, K.square(ip1 - center[:,0]).shape)

    # outer loss
    input_other = Input(shape=(categories - 1, ))
    other_centers = Centers(input_other)
    outer_loss = Lambda(lambda x : K.sum(K.square(x[0][:, 0] - x[1]), axis = 1, keepdims=True), name='outer_loss')([other_centers, tmp])
    # print('*******************', other_centers.shape, ip1.shape, outer_loss.shape, (other_centers - ip1).shape)

    # build model
    model = Model(inputs=[input, input_target], outputs=[predictions, inner_loss])
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
                  loss=["categorical_crossentropy", lambda y_true, y_pred: y_pred], 
                  loss_weights=[1, lambda_inner], 
                  metrics=['accuracy'])

# train and evaluate
cv_score = []
start_time = time.time()
for i in range(len(Y)):
    print('==========={0} fold=============='.format(i))
    X_train, Y_train = [], []
    Y_train_value, Y_test_value = [], []
    for j in range(len(Y)):
        if i == j:
            X_test = X[i]
            Y_test = Y[i]
            Y_test_value.append(Y_value[i])
        else:
            X_train.append(X[j])
            Y_train.append(Y[j])
            Y_train_value.append(Y_value[j])
    X_train = np.concatenate(X_train, axis = 0)
    Y_train = np.concatenate(Y_train, axis = 0)
    Y_test_value = np.concatenate(Y_test_value, axis = 0)
    Y_train_value = np.concatenate(Y_train_value, axis = 0)
    print('=============shape of data==============')
    print(X_train.shape, Y_train.shape, Y_test_value.shape, Y_train_value.shape)

    if not use_clustering_loss:
        model.fit(X_train, 
                  Y_train, 
                  batch_size = batch_size, 
                  epochs = epochs, 
                  validation_data = (X_train, Y_train),
                  # validation_data = (X_test, Y_test),
                  shuffle = True, 
                  verbose = 2,
                  callbacks = [call_back_evaluation])
        score = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose=0)
        cv_score.append(score[1])
    else:
        random_y_train = np.random.rand(X_train.shape[0],1)
        random_y_test = np.random.rand(X_test.shape[0],1)
        
        model.fit([X_train, Y_train_value],
                  [Y_train, random_y_train], 
                  batch_size = batch_size, 
                  epochs = epochs,
                  verbose = 1, 
                  # validation_data = ([X_test, Y_test_value], [Y_test, random_y_test]),
                  validation_data = ([X_train, Y_train_value], [Y_train, random_y_train]),
                  callbacks = [call_back_evaluation]
                  )
        score = model.evaluate([X_test, Y_test_value], [Y_test, random_y_test], batch_size = batch_size, verbose=0)
        print(score)
        cv_score.append(score[3])
    print('**************validation accuracy of fold {0}:{1}******************'.format(i+1, cv_score[-1]))
    print('**************curr average accuracy {0}******************'.format(np.mean(cv_score)))
    # release the memory of GPU taken by the model 
    # K.clear_session()

logging.info('model layers \n {0}'.format(model.summary()))
logging.info('time consuming:{0}s'.format(time.time() - start_time))
logging.info('k-fold accuracy:{0}'.format(cv_score))
logging.info('average accuracy: {0}'.format(np.mean(cv_score)))
logging.info('######################################################################\n')