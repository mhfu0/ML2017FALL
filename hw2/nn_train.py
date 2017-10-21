#!/usr/bin/env python3

import sys
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction=0.333
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=2
#tf.set_random_seed(1234)

set_session(tf.Session(config=config))

'''
import os, random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
'''

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout

from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam

np.random.seed(7)

X=[]
with open('data/X_train', 'r') as f:
    f.readline()
    for line in f:
        line = line.strip('\n').split(',')
        line = list(map(int,line))
        X.append(line)

X=np.array(X)
XMAX=X.max(axis=0)
X_n=X/XMAX
X=X_n

Y=[]
with open('data/Y_train', 'r') as f:
    f.readline()
    for line in f:
        line = line.strip('\n').split(',')
        line = list(map(int,line))
        Y.append(line)
Y=np.array(Y).astype('int32')

x_train=X[:25000,:].astype('float32')
x_test=X[25000:,:].astype('float32')
print(x_train.shape[0], 'train examples')
print(x_test.shape[0], 'test examples')

y_train=Y[:25000]
y_test=Y[25000:]

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(64,input_shape=(106,)))
model.add(Activation('tanh'))
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dropout(0.2))


model.add(Dense(2))
model.add(Activation('softmax'))
rms=RMSprop()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=rms)

model.fit(x_train,y_train,
          batch_size=32,epochs=10,
          verbose=1)
model.summary()

score=model.evaluate(x_test,y_test,verbose=1)
print(score)

testX=[]
with open('data/X_test','r') as f:
    f.readline()
    for line in f:
        line = line.strip('\n').split(',')
        line = list(map(int,line))
        testX.append(line)

testX=np.array(testX)
testX=testX/XMAX

testY=model.predict_classes(testX, verbose=1)

with open(sys.argv[1], 'w') as f:
    f.write('id,label\n')
    for i, l in enumerate(testY):
        f.write('%d,%d\n' % ((i+1),l))

