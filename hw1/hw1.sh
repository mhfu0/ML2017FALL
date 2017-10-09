#!/bin/bash

#./train_linear.py ./data/train.csv ./model/model_linear.out
./test_linear.py ./model/model_linear.out $1 > $2
