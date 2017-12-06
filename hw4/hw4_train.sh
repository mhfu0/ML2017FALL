#!/bin/bash

python3 ./rnn_train.py $1 $2
python3 ./semi_train.py $1 $2