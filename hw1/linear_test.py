#!/usr/bin/env python3
# Linear regression implementation for baseline
# usage: ./linear_test.py [model] [testing]

import sys
import numpy as np
import pandas as pd

# Default settings
feature_idx = set(range(18))   
num_feature = len(feature_idx)
num_feature_t = 18

feature_dropped_idx = set(range(num_feature_t))-feature_idx

def main(argv):
    # y = b + sum(w_i*x_i)
    b = 0.0
    w = np.array(0.0)
    
    # read model parameters
    with open(argv[1], 'r') as model:
        # Read feature settings
        line = model.readline().strip('\n').split(',')
        feature_idx = set(map(int, line))
        feature_dropped_idx = set(range(num_feature_t))-feature_idx
        
        # Read model parameters
        line = model.readline().strip('\n').split(',')
        line = list(map(float, line))
        b = line[0]
        w = np.array(line[1:])
        
        num_w = w.size
    
    print('id,value')   # fisrt line
    
    # read testing data
    test_df = pd.read_csv(argv[2], header=None)
    
    # replace 'NR' as 0.0
    test_df = test_df.replace({'NR':'0.0'})
    
    num_test_entry = len(test_df) // num_feature_t
    for n in range(num_test_entry):
        df = test_df[(n*num_feature_t):((n+1)*num_feature_t)]
        id_name = df.iloc[0,0]
        
        df = df.drop(df.columns[0:2], axis=1)
        
        # Drop features
        df = df.drop(df.index[[list(feature_dropped_idx)]])
        
        df = df.astype(float).as_matrix()
        x_data = df.flatten()
        
        y_predict = np.dot(w.T,x_data)+b
        print(id_name+','+str(y_predict))
        

if __name__ == "__main__":
    main(sys.argv)
