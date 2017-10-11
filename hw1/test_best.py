#!/usr/bin/env python3
# Prediction on test set with saved model
# model: 18lin+2quad 25000iter 25%rand_drop
# ./test_best.py [model] [test] > [result]

import sys
import numpy as np

feature_set=[7,9]
prev_hrs=9

def main(argv):
    # Load model parameters
    b = 0.0
    w = []
    with open(argv[1], 'r') as model_f:
        line = model_f.readline()
        data = line.strip('\n').split(',')
        b = float(data[0])
        w = list(map(float, data[1:]))
        
    # Load test.csv
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
        
        print('%s,%.16f' % (idx, y))

if __name__ == '__main__':
    main(sys.argv)
