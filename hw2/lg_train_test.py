#!/usr/bin/env python3
# argv[1]: {0,1}={test,train}
# argv[2]: model path
# argv[3]: X_train
# argv[4]: Y_train
# argv[5]: X_test
# argv[6]: result path

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
    TRAIN = bool(int(sys.argv[1]))

    X=[]
    with open(sys.argv[3], 'r') as f:
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
    with open(sys.argv[4], 'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            Y.append(line)
    Y=np.array(Y).astype('int32')

    testX=[]
    with open(sys.argv[5],'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            testX.append(line)
    testX=np.array(testX)
    #testX=testX/XMAX

    # Normalization
    X, testX = normalize(X, testX)

    # Train
    if TRAIN:
        # TODO: validation set split
        
        # Initialize weight with normal dist.
        num_f = 106  # number of features
        w = np.random.randn(num_f)
        b = 0.0
        
        lr = 1
        ld = 0.0
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
            loss = (sum(loss)+ld*sum(w**2))/len(X)
            #loss = np.mean(loss)
            if ep % 10 == 9:
                sys.stderr.write('%d iteration with loss %f\n' % (ep+1, loss))
            
            # Stop when little improvement
            #if(np.abs(prev_loss-loss) <= 1e-6):
            #    break
            
            # Compute gradients
            #w_grad=np.mean(-1*X*(Y-y), axis=0)
            w_grad=(np.sum(-1*X*(Y-y), axis=0)+2*ld*w)/len(X)  # regularization
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
        
        # Save model
        with open(sys.argv[2], 'w') as f:
            p = [b] + list(w)
            p = list(map(str, p))
            f.write(' '.join(p))
    
    # Test
    if not TRAIN:
        with open(sys.argv[2], 'r') as f:
            line = f.readline()
            line = line.strip('\n').split(' ')
            b = float(line[0])
            w = np.array(line[1:]).astype(float)
            
        y_test = sigmoid(np.dot(testX,w.T)+b)
        y_test = np.round(y_test)
        
        with open(sys.argv[6], 'w') as f:
            f.write('id,label\n')
            for i, l in enumerate(y_test):
                f.write('%d,%d\n' % ((i+1),l))
