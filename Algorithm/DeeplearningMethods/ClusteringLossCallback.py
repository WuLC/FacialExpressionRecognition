import sys

import keras
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
from sklearn.manifold import TSNE

class Evaluation(keras.callbacks.Callback):

    def __init__(self, use_clustering_loss):
        self.use_clustering_loss = use_clustering_loss

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        print(self.validation_data[0].shape, self.validation_data[1].shape)
        print(self.model.input[0].shape)

    def on_epoch_end(self, epoch, logs={}):
        
        print('\n=========')
        print(len(self.validation_data)) #be careful of the dimenstion of the self.validation_data, somehow some extra dim will be included
        print(self.validation_data[0].shape)
        print(self.validation_data[1].shape)
        
        input = self.model.input[0] if self.use_clustering_loss else self.model.input
        feature_layer_model = Model(inputs=input, outputs=self.model.get_layer('feature').output)
        feature_output = feature_layer_model.predict(self.validation_data[0])
        

        if feature_output.shape[1] != 2:
            print('perform tSEN for diemension reduction')
            scaled_feature = TSNE(n_components=2).fit_transform(feature_output)
        else:
            scaled_feature = feature_output

        if self.use_clustering_loss:
            labels = self.validation_data[1].flatten() # already are single value ground truth labels
        else:
            labels = np.argmax(self.validation_data[1],axis=1)
        
        print('shape of scaled feature', scaled_feature.shape)
        visualize(scaled_feature, labels, epoch)
        print('=========')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def visualize(feat, labels, epoch):

    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(7):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    # plt.legend(['angry', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
    XMax = np.max(feat[:,0]) 
    XMin = np.min(feat[:,1])
    YMax = np.max(feat[:,0])
    YMin = np.min(feat[:,1])

    plt.xlim(xmin=XMin,xmax=XMax)
    plt.ylim(ymin=YMin,ymax=YMax)
    plt.text(XMin,YMax,"epoch=%d" % epoch)
    plt.savefig('../images/epoch=%d.png' % epoch)
    plt.draw()
    plt.pause(0.001)
