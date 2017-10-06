#!/usr/bin/env python3
# Linear regression implementation for baseline
# usage: ./linear_train.py [training] [validation] [model_out]

import sys
import numpy as np

num_w = 162 # default is to use all features

def main(argv):

    np.random.seed()
    
    # input: training set
    x_data = []
    y_data = []
    with open(argv[1], 'r') as train_set:
        # Keep feature index information
        feature_idx = train_set.readline()
        
        for line in train_set:
            if '-1' not in line:
                tmp = line.strip('\n').split(',')
                tmp = list(map(float, tmp))
                
                x_data.append(tmp[1:])
                y_data.append(tmp[0])
    
    # x_data: N by I matrix (N=#entries,I=#features)
    # y_data: N-d vector
    x_data = np.array(x_data).astype(float)
    y_data = np.array(y_data).astype(float)
    num_w = x_data.shape[1]

    # randomly determine model parameters
    # L2_regularization:
    # y = b + sum(w*x) + ld*sum(w^2)
    # TODO: wrong

    b = 1
    w = np.random.randn(num_w)/100
    ld = 0.001  # lambda for regularization
    
    # Apply AdaGrad
    # if fixed rate, lr = 0.00000000035 works well
    lr = 1  # learning rate
    iteration = 10000
    
    # lr_b, lr_w are sigmas for AdaGrad
    lr_b = 0.0
    lr_w = np.zeros(num_w)
    
    prev_loss = float('inf')
    
    for k in range(iteration):
        print('iteration:',k)
        
        b_grad = 0.0
        w_grad = list(np.zeros(num_w))  # w_grad.type:list
        
        # define loss function as square error
        # predicted value y from nth data = nth entry of X*w+b
        y_predict = np.dot(x_data,w) + b + ld*np.sum(np.power(w,2)) # latter two terms are scalors
        dy = y_data - y_predict
        
        # check current loss
        loss = (np.sum(dy ** 2) + ld * np.sum(w ** 2))/len(y)
        print('current loss:',loss)
        
        # Stop when little improvement
        if(np.abs(prev_loss-loss) < 1e-4):
            break
        
        # Compute gradients
        b_grad -= np.sum(2*dy)
        for i in range(num_w):
            w_grad[i] -= np.dot(2*dy, x_data.T[i])+2*ld*w[i]
        
        # Compute sigmas for AdaGrad
        lr_b += b_grad ** 2
        lr_w += np.power(np.array(w_grad),2)
        
        # Update W, b
        b -= lr/np.sqrt(lr_b) * b_grad
        for i in range(num_w):
            w[i] = w[i]-lr/np.sqrt(lr_w[i])*w_grad[i]
            
        prev_loss = loss
    
    print('Training ended with total loss:',loss)
    
    # test with validation set
    print('=====validation=====')
    x_data = []
    y_data = []
    with open(argv[2], 'r') as val_set:
        for line in val_set:
            tmp = line.strip('\n').split(',')
            tmp = list(map(float, tmp))
            
            x_data.append(tmp[1:])
            y_data.append(tmp[0])
    
    x_data = np.array(x_data).astype(float)
    y_data = np.array(y_data).astype(float)
    print('len(x_data)=',len(x_data))

    y_predict = np.dot(x_data,w)+b
    dy = y_data - y_predict
    loss = sum(np.power(dy,2))
    rmse = np.sqrt(sum(np.power(dy,2))/len(x_data))
    print('Total loss on validation set:',loss)
    print('RMSE on validation set:',rmse)
    
    with open(argv[3], 'w') as output:
        line = ''
        line += feature_idx
        line += str(b)+','+','.join(list(w.astype(str)))
        output.write(line)

if __name__== "__main__":
    main(sys.argv)
