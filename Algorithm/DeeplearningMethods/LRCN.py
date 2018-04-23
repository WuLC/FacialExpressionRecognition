# -*- coding: utf-8 -*-
# Created on Thu Apr 19 2018 10:12:43
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import time
from collections import deque

import fire
import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Flatten, GRU, Dense, Dropout
from keras import optimizers
from tqdm import tqdm



# global variables
categories = 7
FIX_INPUT_LEN = True
fixed_seq_len = 10

def build_model():
    pretrained_cnn = VGG16(weights='imagenet', include_top=False)
    # pretrained_cnn.trainable = False
    for layer in pretrained_cnn.layers[:-5]: # keep some layers non-trainable (weights will not be updated)
        layer.trainable = False
    input = Input(shape = (224, 224, 3))
    x = pretrained_cnn(input)
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)
    pretrained_cnn = Model(inputs = input, output = x)

    input_shape = (None, 224, 224, 3) # (seq_len, width, height, channel)
    model = Sequential()
    model.add(TimeDistributed(pretrained_cnn, input_shape=input_shape))
    model.add(GRU(1024, kernel_initializer='orthogonal', bias_initializer='ones', dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(categories, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5),
                metrics=['accuracy'])
    return model


def load_sample(img_dir, fixed_seq_len = None):
    label = int(img_dir.split('/')[-2].split('_')[0]) - 1
    img_names = sorted(os.listdir(img_dir))
    imgs = []
    if FIX_INPUT_LEN: # extract certain length of sequence
        block_len = round(len(img_names)/fixed_seq_len)
        idx = len(img_names) - 1
        tmp = deque()
        for _ in range(fixed_seq_len):
            tmp.appendleft(img_names[idx])
            idx = max(idx-block_len, 0)
        img_names = tmp
    for img_name in img_names:
        img_path = img_dir + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        imgs.append(x)
    imgs = np.array(imgs)
    label = np_utils.to_categorical(label, num_classes=categories)
    if not FIX_INPUT_LEN:
        imgs = np.expand_dims(imgs, axis=0)
        label = label.reshape(-1, categories)
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


def train():
    data_dir = 'F:/FaceExpression/TrainSet/CK+/10_fold/'
    scores = []
    start_time = time.time()
    epochs = 100
    val_fold = 9
    model = build_model()
    if FIX_INPUT_LEN:
        batch_size = 20
        X, Y = load_fix_len_dataset(data_dir)
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
        for epoch in tqdm(range(epochs)):
            model.fit(X_train, Y_train, batch_size= batch_size, epochs=1)
            s = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose=0)[1]
            scores.append(s)
            print('val accuracy for {0} epoch:{1}'.format(epoch, s))
    else:
        dataset = load_var_len_dataset(data_dir)
        for epoch in tqdm(range(epochs)):
            for i in range(10):
                if i != val_fold:
                    for x, y in dataset[i]:
                        model.fit(x, y, batch_size=1, epochs=1,verbose=0)
            s = evaluate_var_len_data(model, dataset[val_fold])
            scores.append(s)
            print('val accuracy for {0} epoch:{1}'.format(epoch, s))
    print(scores)
    print('time consuming for {0} epochs: {1}s'.format(epochs, time.time()-start_time))


if __name__ == '__main__':
    fire.Fire()