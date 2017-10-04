#!/usr/bin/env python3
# Linear regression implementation for baseline
# usage: ./linear_test.py [model] [testing_set]

import sys
import numpy as np
import pandas as pd

num_feature = 18
num_prev_hour = 9

def main(argv):
    # y = b + sum(w_i*x_i)
    b = 0.0
    w = np.array(0.0)
    
    # read model parameters
    with open(argv[1], 'r') as model:
        line = model.readline().strip('\n').split(',')
        line = list(map(float, line))
        b = line[0]
        w = np.array(line[1:])
        
        num_w = w.size
    
    print('id,value')   # fisrt line
    
    # read testing data
    test_df = pd.read_csv(argv[2], header=None)
    
    # replace 'NR' as 0
    test_df = test_df.replace({'NR':'0.0'})
    
    num_test_entry = len(test_df) // num_feature
    for n in range(num_test_entry):
        df = test_df[(n*num_feature):((n+1)*num_feature)]
        id_name = df.iloc[0,0]
        
        df = df.drop(df.columns[0:2], axis=1)
        df = df.astype(float).as_matrix()
        x_data = df.flatten()
        
        y_predict = np.dot(w.T,x_data)+b
        print(id_name+','+str(y_predict))
        

if __name__ == "__main__":
    main(sys.argv)
