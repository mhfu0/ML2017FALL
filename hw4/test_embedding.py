# -*- coding: utf-8 -*-
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
from keras.models import Model, load_model
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

def load_data(path):
    text = []
    with open(path, 'r') as f:
        f.readline()
        for line in f:
            if line == '':
                break
            line = line.strip('\n').split(',')
            text.append(line[1])
    return text

def isascii(s):
    return len(s) == len(s.encode())

def trim(text_list, threshold=2):
    result = []
    for _, text in enumerate(text_list):
        grouping = []
        for _, g in itertools.groupby(text):
            grouping.append(list(g))
        r = ''.join([g[0] if len(g)<threshold else g[0]*threshold for g in grouping])
        result.append(r)
    return result

def remove_special_chara(labels):
    # Borrowed this from https://github.com/chenxinpeng/S2VT/blob/master/model_RGB.py
    labels = map(lambda x: x.replace('.', ''), labels)
    labels = map(lambda x: x.replace(',', ''), labels)
    labels = map(lambda x: x.replace('"', ''), labels)
    labels = map(lambda x: x.replace('\n', ''), labels)
    labels = map(lambda x: x.replace('?', ''), labels)
    labels = map(lambda x: x.replace('!', ''), labels)
    labels = map(lambda x: x.replace('\\', ''), labels)
    labels = map(lambda x: x.replace('/', ''), labels)
    labels = map(lambda x: x.replace('-', ''), labels)
    labels = map(lambda x: x.replace('\(', ''), labels)
    labels = map(lambda x: x.replace('\)', ''), labels)  
    
    return list(labels)

def main():
    test_data_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = 'model.h5'
    dict_path = 'dict.pkl'
    
    sys.stderr.write('Loading data...\n')
    sent = load_data(test_data_path)
    
    # Text preprocessing here
    sent = trim(sent)
    
    word2id = {}
    id2word = {}
    vocab_size = -1
    with open(dict_path, 'rb') as d:
        data = pickle.load(d)
        word2id = data[0]
        id2word = data[1]
        vocab_size = data[2]
    
    sys.stderr.write('Preoprocessing data...\n')
    # Text preprocessing
    sent_ = trim(sent, threshold=1)
    #DEFAULT_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces,
    #                   strip_numeric, remove_stopwords, strip_short, stem_text]
    filters = [lambda x: x.lower(), stem_text, strip_numeric]
    sent = [' '.join(preprocess_string(s, filters=filters)) for s in sent_]
    
    # change words into corresponding indices
    for i in range(len(sent)):
        sent[i] = sent[i].lower().split(' ')
        sent[i] = list(map(lambda x: word2id[x] if x in word2id else word2id['<unk>'], sent[i]))

    # Pad each sentence into same length
    max_sent_len = 30

    x_test = []
    for s in sent:
        padded_sent = np.zeros((max_sent_len), dtype=np.int)
        if max_sent_len >= len(s):
            padded_sent[:len(s)] = s[:len(s)]
        else:
            padded_sent[:max_sent_len] = s[:max_sent_len]
        x_test.append(padded_sent)
    x_test = np.array(x_test)
    
    model = load_model(model_path)
    model.summary()
    
    result = model.predict(x_test, verbose=1)
    result = list(result)
    
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        classify = lambda x: 0 if x < 0.5 else 1
        for i, p in enumerate(result):
            f.write('%d,%d\n' % (i, classify(p)))

if __name__ == '__main__':
    main()