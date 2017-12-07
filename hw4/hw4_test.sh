#!/bin/bash

wget -c -O ./model_semi_0.h5 'https://www.dropbox.com/s/hvgx424mwbh1sql/model_semi_0.h5?dl=1'
wget -c -O ./model_semi_1.h5 'https://www.dropbox.com/s/oarf3z0p2dkxjkl/model_semi_1.h5?dl=1'
wget -c -O ./model_semi_2.h5 'https://www.dropbox.com/s/a1jmjzed4qvztuj/model_semi_2.h5?dl=1'
python3 ./rnn_test.py $1 $2