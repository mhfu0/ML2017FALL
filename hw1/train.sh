#!/bin/bash

./preprocess.py data/train.csv train.in validation.in
./linear_train.py train.in validation.in model.out
