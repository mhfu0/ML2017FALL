# CNN image emotion classifier
# argv[0]: cnn_test.py
# argv[1]: train.csv
# argv[2]: test.csv

import sys
import numpy as np
#from PIL import Image

# Settings for nlg workstation
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction=0.333
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=2
tf.set_random_seed(1234)  # for reproducibily
set_session(tf.Session(config=config))

# Settings for reproducibility
import os, random
os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
np.random.seed(15)

# Import Keras package
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils, plot_model

def load_train(path):
    # Return training data in np.array
    f = open(path, 'r')
    f.readline()  # drop column name
    
    x_train=[]
    y_train=[]
    for line in f:
        line = line.strip('\n').split(',')
        y = int(line[0])
        x = list(map(int,line[1].split(' ')))
        
        y_train.append(y)
        x_train.append(x)
            
    f.close()
    return np.array(x_train), np.array(y_train)

def load_test(path):
    # Return training data in np.array
    f = open(path, 'r')
    f.readline()  # drop column names
    
    idx=[]
    x_test=[]
    for line in f:
        line = line.strip('\n').split(',')
        i = line[0]
        x = list(map(int,line[1].split(' ')))
        
        idx.append(i)
        x_test.append(x)
            
    f.close()
    return idx, np.array(x_test)

def normalize1D(x_train, x_test):
    # Normalization on all x's
    x_all = np.concatenate((x_train, x_test)).astype(float)
    mu = sum(x_all) / len(x_all)
    sigma = np.std(x_all, axis=0)
    
    for i in range(len(x_all)):
        x_all[i]=(x_all[i]-mu)/sigma
    return x_all[0:len(x_train)], x_all[len(x_train):]

def hist_eq(arr, number_bins=256):
    # Histogram equalization on single image
    # Get image histogram
    arr = arr.astype(np.float32)
    hist, bins = np.histogram(arr, number_bins, density=True)
    cdf = hist.cumsum() # CDF
    cdf = 255 * cdf / cdf[-1]

    # use linear interpolation of cdf to find new pixel values
    arr_eq = np.interp(arr, bins[:-1], cdf)

    return arr_eq

def reshape_data(x):
    num_data=x.shape[0]
    x_r = x.reshape((num_data,48,48,1))
    return x_r

def main(argv):
    model_path = 'model/model_gen.h5'
    x_train, y_train = load_train(argv[1])
    idx, x_test = load_test(argv[2])    
    
    '''
    # Histogram Equalization
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    for i in range(x_train.shape[0]):
        x_train[i] = hist_eq(np.squeeze(x_train[i]))
    for i in range(x_test.shape[0]):
        x_test[i] = hist_eq(np.squeeze(x_test[i]))
    #x_train /= 255
    #x_test /= 255
    
    # Save as image
    for i in range(len(x_test)):
        arr = np.squeeze(x_test[i])
        arr = arr.astype('uint8')
        im = Image.fromarray(arr)
        im.save('image/train/%.5d.jpg'%i)
    '''
    
    # Normalization
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)    
    x_train, x_test = normalize1D(x_train, x_test)
    
    # Reshape Data
    x_test = reshape_data(x_test)
    
    # Model parameters
    num_data = x_test.shape[0]
    num_classes = 7
    input_shape = (48,48,1)  # 48*48 with single channel
    
    # Load trained model
    sys.stderr.write('Load trained model...\n')
    model = load_model(model_path)
    model.summary()

    y_test = model.predict_classes(x_test, verbose=0)
    print('id,label')
    for i, l in enumerate(y_test):
        print('%s,%d' % (idx[i],l))

if __name__ == '__main__':
    main(sys.argv)