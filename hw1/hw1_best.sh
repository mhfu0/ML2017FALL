#!/bin/bash

#./train_best.py ./data/train.csv ./model/model_best.out
./test_best.py ./model/model_best.out $1 > $2
