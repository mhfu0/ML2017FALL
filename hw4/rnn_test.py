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
    testing_data_path = sys.argv[1]
    output_path = sys.argv[2]

    #model_path = 'model_semi.h5'
    tokenizer_path = 'tokenizer_max.pickle'
    
    # Model settings
    MAX_SEQUENCE_LENGTH = 40
    #MAX_NUM_WORDS = 20000  # containing padding zeros
    MAX_NUM_WORDS = None
    EMBEDDING_DIM = 100
    
    # Load tokenizer
    f = open(tokenizer_path, 'rb')
    tokenizer = pickle.load(f)
    f.close()
    
    # Load testing data
    f = open(testing_data_path, 'r')
    f.readline()
    test_texts = []
    for line in f:
        pair = line.strip('\n').split(',', 1)
        test_texts.append(pair[1])
    f.close()

    # Text preprocessing
    test_texts = trim(test_texts, threshold=2)
    
    #filters = [lambda x: x.lower(), stem_text, strip_numeric, strip_multiple_whitespaces]
    filters = [lambda x: x.lower(), stem_text, strip_multiple_whitespaces]
    tmp = [' '.join(preprocess_string(s, filters=filters)) for s in test_texts]
    test_texts = tmp
    del tmp

    # Text sequence encoding
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    '''
    # Predict with ensemble model
    NUM_ENSEMBLE = 3
    result = np.zeros((len(x_test), 1), dtype=np.int)
    for i in range(3,6):
        model_path = 'model_semi_%d.h5' % (i)
        model = load_model(model_path)
        model.summary()
        
        result += model.predict_classes(x_test, verbose=1)
        del model
    
    result_list = list(np.squeeze(result))
    
    # Output result
    f = open(output_path, 'w')
    f.write('id,label\n')
    for i, y in enumerate(result_list):
        v = 1 if y > 1 else 0
        f.write('%d,%d\n' % (i, v))
    f.close()
    '''

    # Predict with rnn model
    model_path = 'model_semi_3.h5'
    model = load_model(model_path)
    model.summary()
    
    result = model.predict_classes(x_test, verbose=1)
    del model
    
    result_list = list(np.squeeze(result))
    
    # Output result
    f = open(output_path, 'w')
    f.write('id,label\n')
    for i, y in enumerate(result_list):
        f.write('%d,%d\n' % (i, y))
    f.close()
