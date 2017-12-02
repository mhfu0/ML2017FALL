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
    
    # Model settings
    MAX_SEQUENCE_LENGTH = 30
    MAX_NUM_WORDS = 20000  # containing padding zeros
    EMBEDDING_DIM = 100
    
    # Load tokenizer 
    f = open('tokenizer.pickle', 'rb')
    tokenizer = pickle.load(f)
    f.close()
    
    # Load LSTM model
    model_path = 'model_n.h5'
    model = load_model(model_path)
    model.summary()
    
    # Load testing data
    testing_data_path = sys.argv[1]
    f = open(testing_data_path, 'r')
    
    f.readline()
    test_texts = []
    for line in f:
        pair = line.strip('\n').split(',', 1)
        test_texts.append(pair[1])
    f.close()

    # Text preprocessing
    test_texts = trim(test_texts, threshold=1)
    
    filters = [lambda x: x.lower(), stem_text, strip_numeric, strip_multiple_whitespaces]
    tmp = [' '.join(preprocess_string(s, filters=filters)) for s in test_texts]
    test_texts = tmp

    # Text sequence encoding
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Predict with trained model
    result = model.predict_classes(x_test, verbose=1)
    result_list = list(np.squeeze(result))
    
    # Output result
    output_path = sys.argv[2]
    f = open(output_path, 'w')
    f.write('id,label\n')
    for i, y in enumerate(result_list):
        f.write('%d,%d\n' % (i, y))
    f.close()
