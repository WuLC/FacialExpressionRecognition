import os 
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

# set which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='1'

# specify image size, categories, and log file 
height, width = 224, 224
feature_dim = 256
categories = 7
batch_size = 16
epochs = 40
model_base_path = '../models/CK+_VGG16_fine_tunning_epoch({0})_'.format(epochs)
top_model_base_path =  '../models/CK+_bottleneck_fc_model_'
logfile = '../logs/CK+_ClusteringLoss.log'
logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")

use_clustering_loss = True
use_pretrained_embedding = True
call_back_evaluation = Evaluation(use_clustering_loss)

######################### cross validation ##################################

# load and reshape data firstly
feature_pkl_file = '../Datasets/CK+/pkl/unified_10g.pkl'
feature_pkl_file = '../Datasets/CK+/pkl/original_10g.pkl'
# feature_pkl_file = '../Datasets/JAFFE/pkl/original_10g.pkl'
with open(feature_pkl_file, 'rb') as rf:
    X, Y = pickle.load(rf)
    Y_value = Y[:]
for i in range(len(Y)):
    Y[i] = np_utils.to_categorical(Y[i], categories)

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
    # pretrained embedding matrix
    if use_pretrained_embedding:
        pretrained_embedding = './pretrained_embedding.npy'
        if not os.path.exists(pretrained_embedding):
            model = Model(input, tmp)
            func = K.function([model.input, K.learning_phase()], [model.layers[-1].output])
            count, record = {}, {}
            for x, y in zip(X, Y_value):
                output = func([x, 1.0])
                for i in range(len(y)):
                    if y[i] not in count:
                        count[y[i]] = 1
                        record[y[i]] = np.array(output[0][i])
                    else:
                        count[y[i]] += 1
                        record[y[i]] += np.array(output[0][i])
            embedding = []
            for i in range(categories):
                embedding.append(record[i]/count[i])
            np.save(pretrained_embedding, np.array(embedding))
        Centers = Embedding(categories, feature_dim, weights = [np.load(pretrained_embedding)])
    else:
        Centers = Embedding(categories, feature_dim)
    
    lambda_inner, lambda_outer = 0.0005, 0.0000005
    # inner loss
    input_target = Input(shape=(1,)) # single value ground truth labels as inputs
    center = Centers(input_target)
    inner_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True),name='inner')([tmp, center])

    # outer loss
    input_other = Input(shape=(categories - 1, ))
    other_centers = Centers(input_other)
    # outer_loss = Lambda(lambda x : K.sum(K.square(x[0] - x[1][:,0]), axis = 1, keepdims=True), name='outer_loss')([tmp, other_centers])
    outer_loss = Lambda(lambda x : K.sum(K.sum(K.square(K.expand_dims(x[0], axis=1) - x[1][:,:]), axis = 1), axis = 1, keepdims=True), name='outer')([tmp, other_centers])
    
    # build model
    # center loss
    # model = Model(inputs=[input, input_target], outputs=[predictions, inner_loss])
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
    #               loss=["categorical_crossentropy", lambda y_true, y_pred: y_pred], 
    #               loss_weights=[1, lambda_inner], 
    #               metrics=['accuracy'])

    # clustering loss
    model = Model(inputs=[input, input_target, input_other], outputs=[predictions, inner_loss, outer_loss])
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
                  loss=["categorical_crossentropy", lambda y_true, y_pred: y_pred, lambda y_true, y_pred: y_pred], 
                  loss_weights=[1, lambda_inner, lambda_outer], 
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
    Y_train_other_value = np.array([[i for i in range(categories) if i != num] for num in Y_train_value])
    Y_test_other_value = np.array([[i for i in range(categories) if i != num] for num in Y_test_value])

    print('=============shape of data==============')
    print(X_train.shape, Y_train.shape, Y_test_value.shape, Y_train_value.shape)

    if not use_clustering_loss:
        model.fit(X_train, 
                  Y_train, 
                  batch_size = batch_size, 
                  epochs = epochs, 
                  # validation_data = (X_train, Y_train),
                  validation_data = (X_test, Y_test),
                  shuffle = True, 
                  verbose = 2,
                  callbacks = [call_back_evaluation])
        score = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose=0)
        cv_score.append(score[1])
    else:
        random_y_train = np.random.rand(X_train.shape[0],1)
        random_y_test = np.random.rand(X_test.shape[0],1)

        # center loss
        # model.fit([X_train, Y_train_value],
        #           [Y_train, random_y_train],
        #           batch_size = batch_size, 
        #           epochs = epochs,
        #           verbose = 1, 
        #           # validation_data = ([X_test, Y_test_value], [Y_test, random_y_test]),
        #           validation_data = ([X_train, Y_train_value], [Y_train, random_y_train]),
        #           callbacks = [call_back_evaluation]
        #           )
        # score = model.evaluate([X_train, Y_train_value], 
        #                        [Y_train, random_y_train], 
        #                        batch_size = batch_size, 
        #                        verbose=0)
        # clustering loss
        model.fit([X_train, Y_train_value, Y_train_other_value],
                  [Y_train, random_y_train, random_y_train], 
                  batch_size = batch_size, 
                  epochs = epochs,
                  verbose = 1, 
                  # validation_data = ([X_test, Y_test_value], [Y_test, random_y_test]),
                  validation_data = ([X_train, Y_train_value, Y_train_other_value], [Y_train, random_y_train, random_y_train]),
                  callbacks = [call_back_evaluation]
                  )
        score = model.evaluate([X_train, Y_train_value, Y_train_other_value], 
                               [Y_train, random_y_train, random_y_train], 
                               batch_size = batch_size, 
                               verbose=0)
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