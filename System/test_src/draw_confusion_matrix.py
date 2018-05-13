# -*- coding: utf-8 -*-
# Created on Tue Dec 26 2017 10:48:42
# Author: WuLC
# EMail: liangchaowu5@gmail.com

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_confusion_matrix(log_file):
    index = {'angry':0,
             'disgust':1,
             'fear':2,
             'happy':3,
             'neutral':4,
             'sad':5,
             'surprise':6
             }
    true_label, predict_label = [], []
    with open(log_file, 'r') as rf:
        for line in rf:    
            image_name, true, predict, result = line.strip().split()
            true_label.append(index[true])
            predict_label.append(index[predict])

    cnf_matrix = confusion_matrix(true_label, predict_label)
    return cnf_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')


def main():
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    log_file = 'F:/FaceExpression/processed/front_regroup_oneframe/facepp_all.log'
    cm = get_confusion_matrix(log_file)
    plot_confusion_matrix(cm, class_names)

if __name__ == '__main__':
    main()