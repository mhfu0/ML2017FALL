#!/usr/bin/env python3
# argv[2]: X_train
# argv[3]: Y_train
# argv[4]: X_test

import sys
import math, random
import numpy as np

def sigmoid(z):
    res = 1.0 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def shuffle(X, Y):
    order = np.arange(len(X))
    np.random.shuffle(order)
    return (X[order], Y[order])

def normalize(X, testX):
    allX = np.concatenate((X, testX)).astype(float)
    mu = sum(allX) / len(allX)
    sigma = np.std(allX, axis=0)
    
    for i in range(len(allX)):
        allX[i]=(allX[i]-mu)/sigma
    return allX[0:len(X)], allX[len(X):]

if __name__ == '__main__':
    np.random.seed(7)

    X=[]
    with open(sys.argv[2], 'r') as f:
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
    with open(sys.argv[3], 'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            Y.append(line)
    Y=np.array(Y).astype('int32')

    testX=[]
    with open(sys.argv[4],'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            testX.append(line)
    testX=np.array(testX)
    #testX=testX/XMAX
    
    # Normalization
    X, testX = normalize(X, testX)

    # Gaussian distribution parameters
    num_data = X.shape[0]
    N0 = 0
    N1 = 0

    mu0 = np.zeros((106,))
    mu1 = np.zeros((106,))
    for i in range(num_data):
        if Y[i] == 1:
            mu0 += X[i]
            N0 += 1
        else:
            mu1 += X[i]
            N1 += 1
    mu0 /= N0
    mu1 /= N1

    sigma0 = np.zeros((106,106))
    sigma1 = np.zeros((106,106))
    for i in range(num_data):
        if Y[i] == 1:
            sigma0 += np.dot(np.transpose([X[i] - mu0]), [(X[i] - mu0)])
        else:
            sigma1 += np.dot(np.transpose([X[i] - mu1]), [(X[i] - mu1)])
    sigma = (float(N0) / num_data) * (sigma0/N0) + (float(N1) / num_data) * (sigma1/N1)

    sigma_inv = np.linalg.inv(sigma)
    w = np.dot( (mu0-mu1), sigma_inv)
    x = testX.T
    b = (-0.5) * np.dot(np.dot([mu0], sigma_inv), mu0) + (0.5) * np.dot(np.dot([mu1], sigma_inv), mu1) + np.log(float(N0)/N1)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    print('id,label')
    for i,v in enumerate(y_):
        print('%d,%d'%(i+1,v))