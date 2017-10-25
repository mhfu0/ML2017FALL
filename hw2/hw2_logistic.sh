#!/bin/bash
#$1: raw data (train.csv)              $2: test data (test.csv)  
#$3: provided train feature (X_train)  $4: provided train label (Y_train)
#$5: provided test feature (X_test)    $6: prediction.csv

#python3 ./lg_train_test.py 1 model/model_lg $3 $4 $5 $6
python3 ./lg_train_test.py 0 model/model_lg $3 $4 $5 $6