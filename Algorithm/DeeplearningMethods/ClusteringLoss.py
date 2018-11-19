import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import pickle
import time
import numpy as np
import logging

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import np_utils
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD


# specify image size, categories, and log file 
height, width = 224, 224
feature_dim = 2048
categories = 7
batch_size = 32
epochs = 10
model_base_path = './models/CK+_aug_VGG16_fine_tunning_epoch({0})_'.format(epochs)
top_model_base_path =  './models/CK+_bottleneck_fc_model_'
logfile = './logs/CK+_aug_CustomModels.log'
logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")



######################### cross validation ##################################
# load and reshape data firstly
feature_pkl_file = '../Datasets/CK+/pkl/unified_10g.pkl'

with open(feature_pkl_file, 'rb') as rf:
    X, Y = pickle.load(rf)

for i in range(len(Y)):
    Y[i] = np_utils.to_categorical(Y[i], categories)

cv_score = []
extracted_features = []
start_time = time.time()
for i in range(len(Y)):
    print('==========={0} fold=============='.format(i))
    model_path = model_base_path + '{0}.h5'.format(i)
    
    # pretrained VGG16 + pretrained top layer model 
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape = (7, 7, 512)))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(categories, activation='softmax'))
    top_model_weights_path = top_model_base_path + '{0}.h5'.format(i)
    top_model.load_weights(top_model_weights_path)

    # load pretrained model and add the top model
    initial_model = VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')
    for layer in initial_model.layers[:-5]: # keep some layers non-trainable (weights will not be updated)
        layer.trainable = False

    input = Input(shape=(height, width, 3),name = 'image_input')
    output = initial_model(input)
    x = Flatten()(output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(categories, activation = 'softmax')(x)
    model = Model(input, predictions)
    
    # load top model weights
    layers_num = len(top_model.layers)
    for k in range(layers_num):
        model.layers[k - layers_num].set_weights(top_model.layers[k].get_weights())
    
    model.compile(loss='categorical_crossentropy',
              optimizer = SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])

    X_train, Y_train = [], []
    for j in range(len(Y)):
        if i == j:
            X_test = X[i]
            Y_test = Y[i]
        else:
            X_train.append(X[j])
            Y_train.append(Y[j])
    X_train = np.concatenate(X_train, axis = 0)
    Y_train = np.concatenate(Y_train, axis = 0)

    model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test, Y_test), verbose = 2)
    score = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose=0)
    cv_score.append(score[1])
    print('**************validation accuracy of fold {0}:{1}******************'.format(i+1, score[1]))
    print('**************curr average accuracy {0}******************'.format(np.mean(cv_score)))
    # release the memory of GPU taken by the model 
    K.clear_session()

logging.info('model layers \n {0}'.format(model.summary()))
logging.info('time consuming:{0}s'.format(time.time() - start_time))
logging.info('k-fold accuracy:{0}'.format(cv_score))
logging.info('average accuracy: {0}'.format(np.mean(cv_score)))
logging.info('######################################################################\n')