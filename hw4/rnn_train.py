import numpy as np

### Python Standard Libaray ###
import sys, os
import random
import collections
import pickle
import itertools

### Settings for nlg worksation ###
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=2
#config.gpu_options.per_process_gpu_memory_fraction=0.333
tf.set_random_seed(7)
set_session(tf.Session(config=config))
'''

### import Keras modules ###
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import regularizers, initializers

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

### For reproducibilty ###
np.random.seed(7)
os.environ['PYTHONHASHSEED'] = '0'
random.seed(7)

### Other modules for preprocessing ###
from gensim.parsing.preprocessing import *

def trim(text_list, threshold=2):
    result = []
    for _, text in enumerate(text_list):
        grouping = []
        for _, g in itertools.groupby(text):
            grouping.append(list(g))
        r = ''.join([g[0] if len(g)<threshold else g[0]*threshold for g in grouping])
        result.append(r)
    return result

if __name__ == '__main__':
    # Load training data
    labeled_data_path = sys.argv[1]
    unlabeled_data_path = sys.argv[2]
    tokenizer_path = 'tokenizer_max.pickle'
    
    # Model settings
    MAX_SEQUENCE_LENGTH = 40
    #MAX_NUM_WORDS = 20000  # containing padding zeros
    MAX_NUM_WORDS = None
    EMBEDDING_DIM = 100
    
    try:
        x_train = np.load('data/x_train.npy')
        y_train = np.load('data/y_train.npy')
        x_val = np.load('data/x_val.npy')
        y_val = np.load('data/y_val.npy')
        print('Preprocessed data exists... Loaded')
        
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
            print('Tokenizer load from %s...' % (tokenizer_path))
        word_index = tokenizer.word_index
        num_words = len(word_index)
        
    except:
        print('Load labeled raw data...')
        f = open(labeled_data_path, 'r')
        texts = []
        labels = []
        for line in f:
            line = line.strip('\n')
            labels.append(int(line[0]))
            texts.append(line[10:])
        f.close()
        labels = np.array(labels)
        
        print('Load unlabeled raw data...')
        # Load unlabeled data
        f = open(unlabeled_data_path, 'r')
        texts_un = []
        for line in f:
            texts_un.append(line.strip('\n'))
        f.close()
        
        # Text preprocessing
        texts = trim(texts, threshold=2)
        texts_un = trim(texts_un, threshold=2)
        
        #filters = [lambda x: x.lower(), stem_text, strip_numeric, strip_multiple_whitespaces]
        filters = [lambda x: x.lower(), stem_text, strip_multiple_whitespaces]
        tmp = [' '.join(preprocess_string(s, filters=filters)) for s in texts]
        texts = tmp.copy()
        tmp_ = [' '.join(preprocess_string(s, filters=filters)) for s in texts_un]
        texts_un = tmp_
        del tmp, tmp_
        
        # Encode texts into int seq
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='\n\t')
        tokenizer.fit_on_texts(texts+texts_un)
        word_index = tokenizer.word_index
        num_words = len(word_index)
            
        # Save Tokenizer object for testing stage
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
            print('Tokenizer saved as %s' % (tokenizer_path))
        
        sequences = tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        
        '''
        for i in range(3):
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data = data[indices]
            labels = labels[indices]
        '''
    
        # Split validation set
        VALIDATION_SPLIT = 0.2
        num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
        x_train = data[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        x_val = data[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]
        
        np.save('data/x_train.npy', x_train)
        np.save('data/y_train.npy', y_train)
        np.save('data/x_val.npy', x_val)
        np.save('data/y_val.npy', y_val)
    
    model_path = 'model_n.h5'
    
    model = Sequential()
    model.add(Embedding(num_words,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(256, kernel_initializer='truncated_normal', return_sequences=True))
    model.add(LSTM(256, kernel_initializer='truncated_normal'))
    model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01),kernel_initializer='truncated_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.summary()
    
    checkpointer = ModelCheckpoint(filepath=model_path,
                                   monitor='val_acc',save_best_only=True,
                                   verbose=1)
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(x_val, y_val),
              callbacks=[checkpointer])