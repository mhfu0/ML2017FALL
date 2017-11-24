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

def load_labeled_data(path):
    x_train = []
    y_train = []
    with open(path, 'r') as f:
        for line in f:
            if line == '':
                break
            line = line.strip('\n')
            y_train.append(int(line[0]))
            x_train.append(line[10:])
    return x_train, np.asarray(y_train)

def load_unlabeled_data(path):
    x_train_un = []
    with open(path, 'r') as f:
        for line in f:
            if line == '':
                break
            line = line.strip('\n')
            x_train_un.append(line)
    return x_train_un

def isascii(s):
    return len(s) == len(s.encode())

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # Borrowed and modified this from https://github.com/chenxinpeng/S2VT/blob/master/model_RGB.py
    # Original source: NeuralTalk
    sys.stderr.write('preprocessing word counts and creating vocab based on word count threshold %d\n' % (word_count_threshold))
    word_counts = collections.OrderedDict()
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and isascii(w)]
    sys.stderr.write('filtered words from %d to %d\n' % (len(word_counts), len(vocab)))
    vocab_size = len(vocab) + 2
  
    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<unk>'
  
    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<unk>'] = 1
  
    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+2
        ixtoword[idx+2] = w
    word_counts['<pad>'] = nsents
    word_counts['<unk>'] = nsents
  
    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
  
    return wordtoix, ixtoword, vocab_size, bias_init_vector

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
    
    labels = map(lambda x: x.replace(' \' ', '\''), labels)
    labels = map(lambda x: x.strip(' '), labels)
    
    return list(labels)
    
def trim(text_list, threshold=2):
    result = []
    for _, text in enumerate(text_list):
        grouping = []
        for _, g in itertools.groupby(text):
            grouping.append(list(g))
        r = ''.join([g[0] if len(g)<threshold else g[0]*threshold for g in grouping])
        result.append(r)
    return result
    
def shuffle(X, Y):
    order = np.arange(len(X))
    np.random.shuffle(order)
    order = list(order)
    X_ = np.array([b for a,b in sorted(zip(order,X))])
    Y_ = np.array([b for a,b in sorted(zip(order,Y))])
    return (X_, Y_)

def main():
    labeled_data_path = sys.argv[1]
    unlabeled_data_path = sys.argv[2]
    model_path = 'model.h5'
    dict_path = 'dict.pkl'
    
    sys.stderr.write('Loading data...\n')
    x_sent, y_train = load_labeled_data(labeled_data_path)
    #x_sent_un = load_unlabeled_data(unlabeled_data_path)
    
    sys.stderr.write('Preoprocessing data...\n')
    # Text preprocessing
    x_sent = remove_special_chara(x_sent)
    x_sent = trim(x_sent, threshold=1)
    #DEFAULT_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces,
    #                   strip_numeric, remove_stopwords, strip_short, stem_text]
    #filters = [lambda x: x.lower(), stem_text, strip_numeric, strip_multiple_whitespaces]
    #x_sent = [' '.join(preprocess_string(s, filters=filters)) for s in x_sent_]
    
    '''
    MAX_NUM_WORDS = 20000
    #MAX_NUM_WORDS = None
    MAX_SEQUENCE_LENGTH = 40
    
    # Encoding text sequences
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
    tokenizer.fit_on_texts(x_sent)
    word2id = tokenizer.word_index
    x_seq = tokenizer.texts_to_sequences(x_sent)
    
    # Padding sentences with zeros
    x_train = pad_sequences(x_seq, maxlen=MAX_SEQUENCE_LENGTH,
                            padding='pre', truncating='pre')
    
    #vocab_size = min(len(word2id)+1, MAX_NUM_WORDS)
    vocab_size = len(word2id)+1
    max_sent_len = MAX_SEQUENCE_LENGTH
    print('vocab_size = ', vocab_size)
    
    '''
    sys.stderr.write('Preoprocessing data...\n')
    # build word to index dict based on all data
    word2id, id2word, vocab_size, _ = preProBuildWordVocab(x_sent,
                                      word_count_threshold=3)
    with open(dict_path, 'wb') as d:
        pickle.dump([word2id, id2word, vocab_size], d)
    
    # change words into corresponding indices
    for i in range(len(x_sent)):
        x_sent[i] = x_sent[i].lower().split(' ')
        x_sent[i] = list(map(lambda x: word2id[x] if x in word2id else word2id['<unk>'], x_sent[i]))
    
    #for i in range(len(x_sent_un)):
    #    x_sent_un[i] = x_sent_un[i].lower().split(' ')
    #    x_sent_un[i] = list(map(lambda x: word2id[x] if x in word2id else word2id['<unk>'], x_sent_un[i]))
    
    # Pad each sentence into same length
    #max_sent_len = max([len(x) for x in (x_sent + x_sent_un)])
    max_sent_len = 30

    x_train = []
    for s in x_sent:
        padded_sent = np.zeros((max_sent_len), dtype=np.int)
        if max_sent_len >= len(s):
            padded_sent[:len(s)] = s[:len(s)]
        else:
            padded_sent[:max_sent_len] = s[:max_sent_len]
        x_train.append(padded_sent)
    x_train = np.array(x_train)
    
    #x_train_un = []
    #for s in x_sent_un:
    #    padded_sent = np.zeros((max_sent_len), dtype=np.int)
    #    if max_sent_len >= len(s):
    #        padded_sent[:len(s)] = s[:len(s)]
    #    else:
    #        padded_sent[:max_sent_len] = s[:max_sent_len]
    #    x_train_un.append(padded_sent)
    #x_train_un = np.array(x_train_un)
    
    sys.stderr.write('Building models...\n')
    embedding_dim = 128
    latent_dim = 256
    batch_size = 128
    validation_split = 0.1
    epochs = 10
    checkpointer = ModelCheckpoint(filepath=model_path,
                                   monitor='val_loss',save_best_only=True,
                                   verbose=1)
    
    inputs = Input(shape=(max_sent_len,), dtype='int32')
    x = Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  input_length=max_sent_len,
                  mask_zero=True)(inputs)
    x = Bidirectional(LSTM(latent_dim))(x)
    #x = Dense(latent_dim, activation='relu')(x)
    #x = Dropout(0.7)(x)
    x = Dense(latent_dim, activation='relu')(x)
    x = Dense(latent_dim//2, activation='relu')(x)
    x = Dropout(0.7)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    
    for i in range(3):
        x_train, y_train = shuffle(x_train, y_train)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split,
              callbacks=[checkpointer],
              verbose=1)

if __name__ == '__main__':
    main()