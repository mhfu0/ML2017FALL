import numpy as np

### Python Standard Libaray ###
import sys, os
import random
import collections
import pickle
import itertools

### Settings for nlg worksation ###
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=2
#config.gpu_options.per_process_gpu_memory_fraction=0.333
tf.set_random_seed(7)
set_session(tf.Session(config=config))

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
    model_path = 'model_n.h5'
    
    f = open(labeled_data_path, 'r')
    texts = []
    labels = []
    for line in f:
        line = line.strip('\n')
        labels.append(int(line[0]))
        texts.append(line[10:])
    f.close()
    labels = np.array(labels)
    
    # Model settings
    MAX_SEQUENCE_LENGTH = 30
    MAX_NUM_WORDS = 20000  # containing padding zeros
    EMBEDDING_DIM = 100
    
    # Preprocess texts into int seq
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    num_words = min(MAX_NUM_WORDS, len(word_index))
    
    # Save Tokenizer object for testing stage
    with open('tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Split validation set
    VALIDATION_SPLIT = 0.2
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    
    model = Sequential()
    model.add(Embedding(num_words,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(256))
    model.add(Dense(64))
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