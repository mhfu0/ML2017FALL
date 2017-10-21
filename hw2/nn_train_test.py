#!/usr/bin/env python3
# argv[1]: (0,1)=(test,train)
# argv[2]: model path
# argv[3]: X_train
# argv[4]: Y_train
# argv[5]: X_test
# argv[6]: result path

import sys
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction=0.333
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=2
tf.set_random_seed(1234)
set_session(tf.Session(config=config))

import os, random
os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import Dropout

from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam

np.random.seed(15)
TRAIN = bool(int(sys.argv[1]))

X=[]
with open(sys.argv[3], 'r') as f:
    f.readline()
    for line in f:
        line = line.strip('\n').split(',')
        line = list(map(int,line))
        X.append(line)

X=np.array(X)
XMAX=X.max(axis=0)
#X_n=X/XMAX
#X=X_n

Y=[]
with open(sys.argv[4], 'r') as f:
    f.readline()
    for line in f:
        line = line.strip('\n').split(',')
        line = list(map(int,line))
        Y.append(line)
Y=np.array(Y).astype('int32')

x_train=X.astype('float32')
print(x_train.shape, 'train samples')

y_train=Y.astype('int32')
print(y_train.shape, 'labels')
#y_train=np_utils.to_categorical(y_train)

x_test=[]
with open(sys.argv[5],'r') as f:
    f.readline()
    for line in f:
        line = line.strip('\n').split(',')
        line = list(map(int,line))
        x_test.append(line)

x_test=np.array(x_test)
#testX=testX/XMAX

x_all = np.concatenate((x_train, x_test))
mu = sum(x_all) / len(x_all)
sigma = np.std(x_all, axis=0)

for i in range(len(x_all)):
    x_all[i]=(x_all[i]-mu)/sigma
x_train=x_all[:len(x_train)]
x_test=x_all[len(x_train):]

'''
print(x_train[0].tolist())
print(x_test[0].tolist())
'''
if TRAIN:
    model = Sequential()
    model.add(Dense(64,input_shape=(106,)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer='adam')
    model.summary()
    
    model.fit(x_train,y_train,
              batch_size=64,epochs=32,
              validation_split=0.05,
              verbose=1)
    
    #score=model.evaluate(x_test,y_test,verbose=1)
    #print(score)
    
    model.save(sys.argv[2])

if not TRAIN:
    model = load_model(sys.argv[2])
    y_test=model.predict_classes(x_test, verbose=1)
    with open(sys.argv[6], 'w') as f:
        f.write('id,label\n')
        for i, l in enumerate(y_test):
            f.write('%d,%d\n' % ((i+1),l))