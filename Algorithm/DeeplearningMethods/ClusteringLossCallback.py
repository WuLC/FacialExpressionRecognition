import keras
from sklearn.metrics import roc_auc_score
import sys
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np


class Evaluation(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        print(self.validation_data[0].shape, self.validation_data[1].shape, self.validation_data[2].shape)
        print(self.model.input[0].shape)

    def on_epoch_end(self, epoch, logs={}):
        
        print('\n=========')
        print(len(self.validation_data)) #be careful of the dimenstion of the self.validation_data, somehow some extra dim will be included
        print(self.validation_data[0].shape)
        print(self.validation_data[1].shape)
        print('=========')

        input = self.model.input[0]
        labels = self.validation_data[1].flatten() # already are single value ground truth labels
        feature_layer_model = Model(inputs=input, outputs=self.model.get_layer('feature').output)
        feature_output = feature_layer_model.predict(self.model.input[0])
        
        print(feature_output.shape)
        scaled_feature = scaledown(feature_output)
        print(scaled_feature.shape)
        visualize(scaled_feature, labels, epoch)
        
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def euclidean(x0, x1):
    x0, x1 = np.array(x0), np.array(x1)
    d = np.sum((x0 - x1)**2)**0.5
    return d


def scaledown(X, distance=euclidean, rate=0.1, itera=1000, rand_time=10, verbose=1):
    n = len(X)
    # calculate distances martix in high dimensional space
    realdist = np.array([[distance(X[i], X[j]) for j in range(n)] for i in range(n)])
    realdist = realdist / np.max(realdist)  # rescale between 0-1
    minerror = None

    for i in range(rand_time): # search for n times
    #     if verbose:
    #         print("%s/%s, min_error=%s"%(i, rand_time, minerror))
        # initilalize location in 2-D plane randomly
        loc = np.array([[np.random.random(), np.random.random()] for i in range(n)])

        # start iterating
        lasterror = None
        for m in range(itera):

            # calculate distance in 2D plane
            fakedist = np.array([[np.sum((loc[i] - loc[j])**2)**0.5 for j in range(n)] for i in range(n)])

            # calculate move step
            movestep = np.zeros_like(loc)
            total_error = 0
            for i in range(n):
                for j in range(n):                
                    if realdist[i, j] <= 0.01: continue               
                    error_rate = (fakedist[i, j] - realdist[i, j]) / fakedist[i, j]                
                    movestep[i, 0] += ((loc[i, 0] - loc[j, 0]) / fakedist[i, j])*error_rate
                    movestep[i, 1] += ((loc[i, 1] - loc[j, 1]) / fakedist[i, j])*error_rate
                    total_error += abs(error_rate)

            if lasterror and total_error > lasterror: break  # stop iterating if error becomes worse
            lasterror = total_error

            # update location
            loc -= rate*movestep

        # save best location
        if minerror is None or lasterror < minerror:
            minerror = lasterror
            best_loc = loc
    return best_loc


def visualize(feat, labels, epoch):

    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(7):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['angry', '1', '2', '3', '4', '5', '6'], loc = 'upper right')
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
