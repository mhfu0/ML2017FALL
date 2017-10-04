#!/usr/bin/env python3
# Data Pre-procession
# usage ./preprocess.py [input] [training_set] [validation_set]

import sys
import numpy as np
import pandas as pd

num_feature = 18
num_prev_hour = 5
day_per_month = 20

# PM2.5 values are on the 9th row of each day
pm25_index = 9

def main(argv):
    train_df = pd.read_csv(argv[1])
    train_out = open(argv[2],'w')
    val_out = open(argv[3],'w')
    
    # replace 'NR' as 0
    train_df = train_df.replace({'NR':'0.0'})
    
    # consider no cross-date relation
    num_day = len(train_df) // num_feature
    for d in range(num_day):

        day_df = train_df[d*num_feature:(d+1)*num_feature]
        day_df = day_df.drop(day_df.columns[0:3], axis=1)
        
        for hr in range(24-num_prev_hour):
        # keep last-hour PM2.5 value
        
            df_prev = day_df.iloc[:,hr:(hr+num_prev_hour)]
            df_prev = df_prev.as_matrix()
            x = list(df_prev.flatten())
            
            y_hat = day_df.iloc[:,hr+num_prev_hour].iloc[pm25_index]
        
            # output processed data in form:
            # each line starts with y_hat, followed by x's seperated by comma
            
            outline = y_hat+','+','.join(x)+'\n'
            # take data of last day of every month as validation set
            if d % day_per_month != day_per_month-1:
                train_out.write(outline)
            else:
                val_out.write(outline)

if __name__ == "__main__":
    main(sys.argv)
