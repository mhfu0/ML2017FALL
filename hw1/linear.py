#!/usr/bin/env python3
# Linear regression implementation for baseline
# usage: ./linear.py [training_set] [model_out]

import sys
import numpy as np

num_w = 162

def main(argv):

    np.random.seed()
    
    # input: training set
    x_data = []
    y_data = []
    with open(argv[1], 'r') as train_set:
        for line in train_set:
            if '-1' not in line:
                tmp = line.strip('\n').split(',')
                tmp = list(map(float, tmp))
                
                x_data.append(tmp[1:])
                y_data.append(tmp[0])
    
    x_data = np.array(x_data).astype(float)
    y_data = np.array(y_data).astype(float)
    
    print(x_data.shape)

    # randomly determine model parameters
    # y = b + sum(w*x) + [regularization]
    b = 1
    w = np.random.randn(num_w)/1000
    ld = 1  # lambda for regularization
    
    # Apply AdaGrad (fixed rate lr = 0.00000000035 works well)
    lr = 1  # learning rate
    iteration = 100000
    
    lr_b = 0.0
    lr_w = np.zeros(num_w)
    
    for k in range(iteration):
        print('iteration:',k)
        
        b_grad = 0.0
        w_grad = list(np.zeros(num_w))  # w_grad.type:list
        
        # define loss function as square error
        # predicted value y from nth data = nth entry of X*w+b
        y_predict = np.dot(x_data,w)+b
        dy = y_data - y_predict
        
        # check current loss
        loss = sum(np.power(dy,2))
        print('current loss:',loss)
        
        b_grad -= np.sum(2*dy)
        for i in range(num_w):
            w_grad[i] -= np.sum(np.multiply(2*dy,x_data.T[i]))
            
        lr_b += b_grad ** 2
        lr_w += np.power(np.array(w_grad),2)
        
        # Update W, b
        b -= lr/np.sqrt(lr_b) * b_grad
        for i in range(num_w):
            w[i] = w[i]-lr/np.sqrt(lr_w[i])*w_grad[i]
    
    with open(argv[2], 'w') as output:
        line = str(b)+','+','.join(list(w.astype(str)))
        output.write(line)

if __name__== "__main__":
    main(sys.argv)
