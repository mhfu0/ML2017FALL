#!/usr/bin/env python3

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
    
    # Normalization
    X, testX = normalize(X, testX)
    
    # TODO: validation set split
    
    # Initialize weight with normal dist.
    num_f = 106  # number of features
    w = np.random.randn(num_f)
    b = 0.0
    
    lr = 1
    ada_w = np.zeros(len(w))
    ada_b = 0.0
    
    epochs = 1000
    prev_loss = float('inf')
    
    # Gradient descent with adagrad
    for ep in range(epochs):
        X, Y = shuffle(X, Y)
    
        b_grad = 0.0
        w_grad = np.zeros(num_f)
        
        # Set loss function as cross entropy
        # output y from nth data = sigmoid(nth entry of X*w+b)
        y = sigmoid(np.dot(X,w.T)+b)
        y = y[:,np.newaxis]
        loss = (-1)*((Y*np.log(y))+((1-Y)*np.log(1-y)))
        #print(loss)
        loss = sum(loss)
        sys.stderr.write('%d iteration with loss %f\n' % (ep+1, loss))
        
        # Stop when little improvement
        if(np.abs(prev_loss-loss) <= 1e-6):
            break
        
        # Compute gradients
        w_grad=np.mean(-1*X*(Y-y), axis=0)
        b_grad=np.mean(-1*(Y-y))
        
        #w -= lr*w_grad
        #b -= lr*b_grad
        
        # Compute AdaGrad terms
        ada_b += b_grad ** 2
        ada_w += w_grad ** 2
        
        # Update W, b
        b -= lr/np.sqrt(ada_b) * b_grad
        for i in range(len(w)):
            w[i] -= lr/np.sqrt(ada_w[i]) * w_grad[i]        
            
        prev_loss = loss
        
    sys.stderr.write('Training ended with total loss: %f\n' % loss)
    
    # Test
    res = sigmoid(np.dot(testX,w.T)+b)
    res = np.round(res)
    
    print('id,label')
    for i,v in enumerate(res):
        print('%d,%d'%(i+1,v))
