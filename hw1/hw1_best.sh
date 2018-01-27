#!/bin/bash

#python3 src/train_best.py data/train.csv model/model_best
python3 src/test_best.py model/model_best $1 > $2
