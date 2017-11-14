# CNN image emotion classifier
# argv[0]: cnn_train.py
# argv[1]: train.csv

import sys
import gc
import numpy as np
#from PIL import Image

'''
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
'''

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
from keras.layers import BatchNormalization

from keras.utils import np_utils, plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

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
    '''
    # Full-scale stretch on histogram
    r_min = min(arr_eq)
    r_max = max(arr_eq)
    mapping = lambda x: (number_bins-1)*(x-r_min)/(r_max-r_min) if (x>=r_min and x<=r_max) \
                          else (0.0 if x<r_min else (number_bins-1))
    arr_eq = np.array(list(map(mapping ,arr_eq)))
    '''
    return arr_eq

def reshape_data(x):
    num_data=x.shape[0]
    x_r = x.reshape((num_data,48,48,1))
    return x_r

def shuffle(X, Y):
    order = np.arange(len(X))
    np.random.shuffle(order)
    return (X[order], Y[order])

def main(argv):
    gc.collect()

    model_path = 'model_.h5'
    x_train, y_train = load_train(argv[1])
    #idx, x_test = load_test(argv[2])    
    
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
    for i in range(len(x_train)):
        arr = np.squeeze(x_train[i])
        arr = arr.reshape((48,48)).astype('uint8')
        im = Image.fromarray(arr)
        im.save('image/%.5d.jpg'%i)
    
    # Normalization
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train, x_test = normalize1D(x_train, x_test)
    '''
    
    # Reshape Data
    x_train = reshape_data(x_train)
    
    # Process label into one-hot data
    y_train = np_utils.to_categorical(y_train, num_classes=7)
    
    # Model parameters
    num_data = x_train.shape[0]
    num_classes = 7
    input_shape = (48,48,1)  # 48*48 with single channel
    kernel_size = (3,3)
    pool_size = (2,2)
    
    # Build sequential NN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=kernel_size,
                            padding='same', activation='relu',
                            input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=kernel_size,
                            padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=kernel_size,
                            padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    
    for i in range(3):
        model.add(Conv2D(64, kernel_size=kernel_size,
                         padding='same', activation='relu'))
        model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=pool_size))
    
    for i in range(3):
        model.add(Conv2D(128, kernel_size=kernel_size,
                         padding='same', activation='relu'))
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    # Training parameters
    epochs = 100
    batch_size = 32
    checkpointer = ModelCheckpoint(filepath=model_path,
                                   monitor='val_loss',save_best_only=True,
                                   verbose=1)
    earlystopping = EarlyStopping(monitor='val_loss',
                                  patience=10, verbose=1)
                                  
    # Shuffle data n times
    for n in range(9):
        x_train, y_train = shuffle(x_train, y_train)

    # Split validation data manually
    validation_split=0.2
    val_idx=int(len(x_train)*(1-validation_split))
    x_val=x_train[val_idx:]
    y_val=y_train[val_idx:]
    x_train=x_train[:val_idx]
    y_train=y_train[:val_idx]

    # Augmentation on image data
    train_datagen = ImageDataGenerator(rotation_range=20,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       #rescale=1./255,
                       horizontal_flip=True,
                       shear_range=0.1,
                       zoom_range=0.25,
                       #featurewise_center=True,
                       #zca_whitening=True,
                       #fill_mode='nearest'
                       )
    #train_datagen.fit(x_train)
    
    try:
        '''
        model.fit(x_train, y_train,
                  batch_size=batch_size,epochs=epochs,
                  validation_data=(x_val, y_val),
                  callbacks=[checkpointer],
                  verbose=1)
        '''          
        model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=64),
                  steps_per_epoch=x_train.shape[0]//16+4, epochs=epochs,
                  #validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
                  #validation_steps=x_val.shape[0]//8,
                  validation_data=(x_val, y_val),
                  callbacks=[checkpointer],
                  verbose=1)

    except:
        #model.save(model_path)
        del model
        gc.collect()
    
    del model
    #model = load_model(model_path)
    #model.save(model_path)
    gc.collect()

if __name__ == '__main__':
    main(sys.argv)
