# -*- coding: utf-8 -*-
# Created on Thu Apr 19 2018 10:12:43
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import time
import gc
from collections import deque
from datetime import datetime

import fire
import visdom
import numpy as np
from tqdm import tqdm
from keras.preprocessing import image
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Flatten, GRU, Dense, Dropout, LSTM
from keras import optimizers
from keras import backend as K


class Configuration:
    def __init__(self):
        self.dataset = 'CKPlus'
        self.vis = visdom.Visdom(env = self.dataset)

        # input and output
        self.categories = 7
        self.fix_input_len = True
        self.fixed_seq_len = 8
        self.img_size = (224, 224, 3)
        self.input_size = (None, 224, 224, 3) # (seq_len, width, height, channel)
        
        # training
        self.batch_size = 10
        self.epochs = 100
        # self.optimizer = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5)


CONF = Configuration()


def build_model():
    pretrained_cnn = VGG19(weights='imagenet', include_top=False)
    # pretrained_cnn.trainable = False
    for layer in pretrained_cnn.layers[:-4]: # keep some layers non-trainable (weights will not be updated)
        layer.trainable = False
    input = Input(shape = CONF.img_size)
    x = pretrained_cnn(input)
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)
    pretrained_cnn = Model(inputs = input, output = x)

    input_shape = CONF.input_size
    model = Sequential()
    model.add(TimeDistributed(pretrained_cnn, input_shape=input_shape))
    model.add(GRU(1024, kernel_initializer='orthogonal', bias_initializer='ones', dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(CONF.categories, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',
                 optimizer = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5), # CONF.optimizer,
                 metrics=['accuracy'])
    return model


def load_sample(img_dir, fixed_seq_len = None):
    label = int(img_dir.split('/')[-2].split('_')[0]) - 1
    img_names = sorted(os.listdir(img_dir))
    imgs = []
    if CONF.fix_input_len: # extract certain length of sequence
        block_len = round(len(img_names)/fixed_seq_len)
        idx = len(img_names) - 1
        tmp = deque()
        for _ in range(CONF.fixed_seq_len):
            tmp.appendleft(img_names[idx])
            idx = max(idx-block_len, 0)
        img_names = tmp
    for img_name in img_names:
        img_path = img_dir + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        imgs.append(x)
    imgs = np.array(imgs)
    label = np_utils.to_categorical(label, num_classes= CONF.categories)
    if not CONF.fix_input_len:
        imgs = np.expand_dims(imgs, axis=0)
        label = label.reshape(-1, CONF.categories)
    return imgs, label


def load_var_len_dataset(data_dir):
    dataset = []
    for i in range(1, 11):
        fold = []
        group_dir = data_dir + 'g{0}/'.format(i)
        for d in os.listdir(group_dir):
            img_dir = group_dir + d + '/'
            x, y = load_sample(img_dir)
            fold.append((x,y))
        dataset.append(fold)
    return dataset


def load_fix_len_dataset(data_dir, fixed_seq_len = 6):
    X, Y = [], []
    for i in range(1, 11):
        tx, ty = [], []
        group_dir = data_dir + 'g{0}/'.format(i)
        for d in os.listdir(group_dir):
            img_dir = group_dir + d + '/'
            x, y = load_sample(img_dir, fixed_seq_len=fixed_seq_len)
            tx.append(x)
            ty.append(y)
        #print(np.array(tx).shape, np.array(ty).shape)
        X.append(np.array(tx))
        Y.append(np.array(ty))
    return X, Y


def evaluate_var_len_data(model, val_data):
    score, count = 0, 0
    for x, y in val_data:
        score += model.evaluate(x, y, batch_size = 1, verbose=0)[1]
        count += 1
    # print('===========val accuracy: {0}'.format(score/count))
    return score/count


def visualize(train_accuracy, val_accuracy, title='default'):
    assert len(train_accuracy) == len(val_accuracy)
    x_epoch = list(range(len(train_accuracy)))
    train_acc = dict(x=x_epoch, y=train_accuracy, type='custom', name='train')
    val_acc = dict(x=x_epoch, y=val_accuracy, type='custom', name='val')
    layout=dict(title=title, xaxis={'title':'epochs'}, yaxis={'title':'accuracy'})
    data = [train_acc, val_acc]
    CONF.vis._send({'data':data, 'layout':layout, 'win':title})


def train():
    data_dir = 'F:/FaceExpression/TrainSet/CK+/10_fold/'
    start_time = time.time()
    best_result = []
    val_fold = 1
    if CONF.fix_input_len:
        X, Y = load_fix_len_dataset(data_dir, CONF.fixed_seq_len)
        for val_fold in range(0, 10):
            train_accuracy, val_accuracy = [], []
            model = build_model()
            X_train, Y_train = [], []
            for i in range(10):
                if i == val_fold:
                    X_test = X[i]
                    Y_test = Y[i]
                else:
                    X_train.append(X[i])
                    Y_train.append(Y[i])
            X_train = np.concatenate(X_train, axis = 0)
            Y_train = np.concatenate(Y_train, axis = 0)
            print(X_train.shape, Y_train.shape)
            print(X_test.shape, Y_test.shape)       
            for epoch in tqdm(range(CONF.epochs)):
                model.fit(X_train, Y_train, batch_size= CONF.batch_size, epochs=1)
                train_acc = model.evaluate(X_train, Y_train, batch_size = CONF.batch_size, verbose=0)[1]
                val_acc = model.evaluate(X_test, Y_test, batch_size = CONF.batch_size, verbose=0)[1]
                train_accuracy.append(train_acc)
                val_accuracy.append(val_acc)
                if (epoch+1) % 5 == 0:
                    visualize(train_accuracy, val_accuracy, title = 'Fold{0}_FIX{1}_VGG16_FintTune3_Dense1024_LSTM2048_bs{2}'.format(\
                    val_fold, CONF.fixed_seq_len, CONF.batch_size))
            best_result.append(max(val_accuracy))
            print(best_result)
            # release the memory of GPU taken by the model of last fold
            K.clear_session()
            gc.collect() 
    else:
        dataset = load_var_len_dataset(data_dir)
        for epoch in tqdm(range(CONF.epochs)):
            for i in range(10):
                if i != val_fold:
                    for x, y in dataset[i]:
                        model.fit(x, y, batch_size=1, epochs=1,verbose=0)
            val_acc = evaluate_var_len_data(model, dataset[val_fold])
            val_accuracy.append(val)
            print('val accuracy for {0} epoch:{1}'.format(epoch, s))
        
    print('[{0}] time consuming for {1} epochs: {2}s'.format(str(datetime.now()), CONF.epochs, int(time.time()-start_time)))


if __name__ == '__main__':
    fire.Fire()