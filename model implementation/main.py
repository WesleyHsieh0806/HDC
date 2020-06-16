# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd
from math import floor
import sys
import HDC_mulpc
import HDC
import math
import time


def main():
    train_x_path = r'd:\電子書\專題\HDC\model implementation\HDC (for undergrduate)\Data\MNIST\MNIST_CNN_4_feature.csv'
    train_y_path = r'D:\電子書\專題\HDC\model implementation\HDC (for undergrduate)\Data\MNIST\MNIST_CNN_4_label.csv'

    # read the file to obtain the data
    with open(train_x_path) as f:
        x = np.array([line.strip().split(',')[0:] for line in f]).astype(float)
    with open(train_y_path) as f:
        y = np.array([line.strip().split(',')[0]
                      for line in f]).astype(float).astype(int).reshape(-1, 1)

    # contruct HDC model
    # dim class feature
    MNIST = HDC_mulpc.HDC(10000, 10, len(x[0]), 21)

    # split train set and validation set
    train_x, train_y, validation_x, validation_y = split_into_validation(
        x, y, 0.8)

    # training and acquire prediction array
    MNIST.train(train_x, train_y)
    ypred = MNIST.test(validation_x)

    # print the accuracy
    acc = MNIST.accuracy(y_pred=ypred, y_true=validation_y)
    print(acc)

    # print result to csv file
    MNIST.result_to_csv('./result.csv')


def split_into_validation(x, y, ratio=0.8):
    '''split x,y into train data and validation data'''
    border = math.floor(ratio*len(x))
    train_x = x[0:border]
    train_y = y[0:border]
    validation_x = x[border:]
    validation_y = y[border:]
    return train_x, train_y, validation_x, validation_y


if __name__ == "__main__":
    main()
