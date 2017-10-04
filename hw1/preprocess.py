#!/usr/bin/env python3
# Data Pre-procession
# usage: ./preprocess.py [input] [training] [validation]

### File Format ### 
#
# [input]:       train.csv
# [training]:    1st line reads used feature-row indices,
#                followed by lines starts with ground-true
#                y_hat, proceed with feature x's
# [validation]:  lines starts with ground-true y proceed with
#                feature x's
#
###################
          
import sys
import numpy as np
import pandas as pd

feature_idx = set([9,10])   # Set which features to use
num_feature = len(feature_idx)
num_feature_t = 18

feature_dropped_idx = set(range(18))-feature_idx  # type:set

num_prev_hour = 9
day_per_month = 20

# PM2.5 values are on the 9th row of each day
pm25_index = 9

def main(argv):
    train_df = pd.read_csv(argv[1])
    train_out = open(argv[2],'w')
    val_out = open(argv[3],'w')
    
    # Write used feature indices on first line
    train_out.write(','.join([str(i) for i in feature_idx])+'\n')
    
    # Replace 'NR' as 0.0
    train_df = train_df.replace({'NR':'0.0'})
    
    # Consider no cross-date relation
    num_day = len(train_df) // num_feature_t
    for d in range(num_day):

        day_df = train_df[d*num_feature_t:(d+1)*num_feature_t]
        day_df = day_df.drop(day_df.columns[0:3], axis=1)
        
        # For x in 0~14th hour, y = [9~23rd PM2.5's]
        pm25 = list(day_df.iloc[pm25_index,num_prev_hour:24])
        
        # Drop specified features
        day_df = day_df.drop(day_df.index[[list(feature_dropped_idx)]])
        
        for hr in range(24-num_prev_hour):
        
            df_prev = day_df.iloc[:,hr:(hr+num_prev_hour)]
            df_prev = df_prev.as_matrix()
            x = list(df_prev.flatten())
            
            y_hat = pm25[hr]
        
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
