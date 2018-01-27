#!/bin/bash

#python3 src/train_linear.py data/train.csv model/model_linear
python3 src/test_linear.py model/model_linear $1 > $2
