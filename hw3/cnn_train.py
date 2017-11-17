# CNN image emotion classifier
# argv[0]: cnn_train.py
# argv[1]: train.csv

import sys
import numpy as np

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
    model_path = 'model.h5'
    x_train, y_train = load_train(argv[1])
    
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
                       horizontal_flip=True,
                       shear_range=0.1,
                       zoom_range=0.25)
                       
    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=64),
              steps_per_epoch=x_train.shape[0]//16+4, epochs=epochs,
              validation_data=(x_val, y_val),
              callbacks=[checkpointer],
              verbose=1)
    
if __name__ == '__main__':
    main(sys.argv)
