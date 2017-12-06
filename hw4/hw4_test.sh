#!/bin/bash

wget -c -O ./model_semi_3.h5 'https://www.dropbox.com/s/v198ulklbgjfyzy/model_semi_3.h5?dl=1'
python3 ./rnn_test.py $1 $2