#!/usr/bin/env python3
# Linear regresssion
# ./train_test_v2.py [train] [test] [model]

import sys
import numpy as np

feature_set=[7,9]
prev_hrs=5
iteration=25000
drop=120
ld = 0.01

def main(argv):
    np.random.seed(1)
    
    raw = []
    with open(argv[1],'r') as train_f:
        train_f.readline()
        for line in train_f:
            line = line.replace('NR', '0.0')
            
            data = line.strip('\n').split(',')
            data = data[3:]
            data = list(map(float, data))
            raw.append(data)
    
    # Reshape as monthly data
    train_data = []
    for m in range(12):
        month_data = []
        
        for f in range(18):
            tmp = []
            
            for d in range(20):
                tmp += raw[360*m+18*d+f]
            
            month_data.append(tmp)
            
        train_data.append(month_data)
    
    # Feature extraction
    x = []
    y_hat = []
    for m in range(12):
        for hr in range(480-prev_hrs):
            y_hat.append(train_data[m][9][hr+prev_hrs])
            
            x.append([])
            for f in range(18):
                # Pick features that we want
                if f in feature_set:
                    x[(480-prev_hrs)*m+hr] += train_data[m][f][hr:hr+prev_hrs]
   
    # Concatenate with some quadratic terms
    for n in range(len(x)):
        row = x[n]
        x[n].append(row[-1] ** 2)
        x[n].append(row[-2] ** 2)
    
    # Seperate some data for validation
    x_val = []
    y_val = []
    
    x_tmp = []
    y_tmp = []
    for n in range(len(x)):
        if n % (480-prev_hrs) < ((480-prev_hrs)-drop):
            x_tmp.append(x[n])
            y_tmp.append(y_hat[n])
        else:
            x_val.append(x[n])
            y_val.append(y_hat[n])
    
    x = np.array(x_tmp)
    y_hat = np.array(y_tmp)
    
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    
    # Regression with gradient descent
    # dim(x)=len(x) by (prev_hrs*# features)
    # dim(y_hat)=len(x)
    
    # y = b + sum(w_Tx)
    # Randomly determine model parameters
    b = 1
    w = np.random.randn(len(x[0]))

    lr = 1
    ada_b = 0.0
    ada_w = np.zeros(len(w))

    prev_loss = float('inf')

    # Gradient descent with adagrad
    for k in range(iteration):
        b_grad = 0.0
        w_grad = np.zeros(len(w))
        
        # Set loss function as square error
        # Predicted value y from nth data = nth entry of X*w+b
        y = np.dot(x,w) + b
        dy = y_hat - y
        
        # Check current loss
        # Apply L2 regularization
        loss = (np.sum(dy ** 2) + ld * np.sum(w ** 2))/len(y)
        sys.stderr.write('%d iteration with loss %f\n' % (k, loss))
        
        # Stop when little improvement
        if(np.abs(prev_loss-loss) < 1e-6):
            break
            
        # Compute gradients
        b_grad -= np.sum(2*dy)
        for i in range(len(w)):
            w_grad[i] -= np.dot( 2*dy, x.T[i] ) + 2*ld*w[i]

        # Compute AdaGrad terms
        ada_b += b_grad ** 2
        ada_w += w_grad ** 2
        
        # Update W, b
        b -= lr/np.sqrt(ada_b) * b_grad
        for i in range(len(w)):
            w[i] -= lr/np.sqrt(ada_w[i]) * w_grad[i]        
            
        prev_loss = loss

    sys.stderr.write('Training ended with total loss: %f\n' % loss)
    
    # Validation
    sys.stderr.write('=====validation=====\n')
    
    y = np.dot(x_val,w) + b
    dy = y_val - y
    loss = sum(np.power(dy,2)) / len(x_val)
    rmse = np.sqrt(sum(np.power(dy,2)) / len(x_val))
    sys.stderr.write('Total loss on validation set: %f\n' % loss)
    sys.stderr.write('RMSE on validation set: %f\n' % rmse)
    
    
    # Save model
    with open(argv[3], 'w') as output:
        line = ''
        line += str(b)+','+','.join(list(w.astype(str)))
        output.write(line)
        
        
    # Prediction on test set
    test_raw = []
    with open(argv[2], 'r') as test_f:
        for line in test_f:
            line = line.replace('NR', '0.0')
            data = line.strip('\n').split(',')
            test_raw.append(data)
    
    print('id,value')
    for i in range(240):
        idx = test_raw[i*18][0]
        
        x = []
        for f in range(18):
            if f in feature_set:
                x += test_raw[i*18+f][2+(9-prev_hrs):11]
        
        x = list(map(float, x))
        row = x
        x.append(row[-1] ** 2)
        x.append(row[-2] ** 2)
        x = np.array(x)
        
        y = np.dot(w,x)+b
        
        print('%s,%f' % (idx, y))

if __name__ == '__main__':
    main(sys.argv)
