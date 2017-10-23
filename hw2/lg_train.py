#!/usr/bin/env python3

import sys
import math, random
import numpy as np

def sigmoid(z):
    res = 1.0 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def normalize(X, testX):
    all = np.concatenate((X, testX))
    mu = sum(all) / len(all)
    sigma = np.std(all, axis=0)
    
    for i in range(len(all)):
        X[i]=(X[i]-mu)/sigma
    return all[0:len(X)], all[len(X):0]

if __name__ == '__main__'

    np.random.seed(7)

    X=[]
    with open('data/X_train', 'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            X.append(line)
    X=np.array(X)
    XMAX=X.max(axis=0)
    #X_n=X/XMAX
    #X=X_n

    Y=[]
    with open('data/Y_train', 'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            Y.append(line)
    Y=np.array(Y).astype('int32')

    testX=[]
    with open('data/X_test','r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            testX.append(line)
    testX=np.array(testX)
    #testX=testX/XMAX
    
    # TODO: validation set split
    
    # initialize weight with normal dist.
    w = np.random.randn(106)
    b = np.random.randn()
    
    lr = 0.1
    
