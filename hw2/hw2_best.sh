#!/bin/bash
#$1: raw data (train.csv)              $2: test data (test.csv)  
#$3: provided train feature (X_train)  $4: provided train label (Y_train)
#$5: provided test feature (X_test)    $6: prediction.csv

# NN model
#python3 ./nn_train_test.py 1 model/model_best.h5 $3 $4 $5 $6
#python3 ./nn_train_test.py 0 model/model_best.h5 $3 $4 $5 $6

# Xgboost with default settings
python3 ./xgb_train_test.py $3 $4 $5 > $6