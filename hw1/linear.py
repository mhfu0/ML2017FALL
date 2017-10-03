#!/usr/bin/env python3
# Linear regression implementation for baseline
# usage: ./linear.py [training_set] [validation_set]

import sys
import numpy as np

num_w = 162

def main(argv):

    np.random.seed(0)
    
    # input: training set
    x_data = []
    y_data = []
    with open(argv[1], 'r') as train_set:
        for line in train_set:
            tmp = line.strip('\n').split(',')
            tmp = list(map(float, tmp))
            
            x_data.append(tmp[1:])
            y_data.append(tmp[0])
    
    x_data = np.array(x_data).astype(float)
    y_data = np.array(y_data).astype(float)

    # randomly determine model parameters
    # y = b + sum(w*x) + [regularization]
    b = 1
    w = np.random.randn(num_w)/1000
    ld = 1  # lambda for regularization
    
    lr = 0.00000000035  # learning rate
    iteration = 10000
    
    for k in range(iteration):
        print('iteration:',k)
        
        b_grad = 0.0
        w_grad = list(np.zeros(num_w))
        
        # define loss function as square error
        # predicted value y from nth data = nth entry of X*w+b
        y_predict = np.dot(x_data,w)+b
        dy = y_data - y_predict
        
        # check current loss
        loss=sum(np.power(dy,2))
        print('current loss:',loss)
        
        b_grad = b_grad - np.sum(2*dy)
        for i in range(num_w):
            w_grad[i] = w_grad[i] - np.sum(np.multiply(2*dy,x_data.T[i]))
            
        # Update W, b
        b = b - lr * b_grad
        w = w - lr * np.array(w_grad)
    
        

if __name__== "__main__":
    main(sys.argv)
