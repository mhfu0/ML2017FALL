#!/bin/bash

#python3 train_linear.py data/train.csv model/model_linear
python3 test_linear.py model/model_linear $1 > $2
