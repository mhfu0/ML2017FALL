import numpy as np

### Python Standard Libaray ###
import sys, os
import random
import collections
import pickle
import itertools

'''
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
'''

### import Keras modules ###
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *
from keras.initializers import *
import keras.backend as K

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# import other utils ###
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    ### For reproducibilty ###
    np.random.seed(7)
    random.seed(7)

    TRAIN = False
    r=39  # random seed
    d = 128  # latent dim

    if TRAIN:
    
        train_data_path = 'train.csv'
        with open(train_data_path, 'r') as f:
            f.readline()
            users = []
            movies = []
            ratings = []
            for line in f:
                line = line.strip('\n').split(',')
                users.append(int(line[1]))
                movies.append(int(line[2]))
                ratings.append(int(line[3]))

        num_users=max(users)
        num_movies=max(movies)

        users = np.array(users)
        movies = np.array(movies)
        ratings = np.array(ratings, dtype=np.float32)
        num_data = len(ratings)

        # validation split
        users_train, users_val, movies_train, movies_val, ratings_train, ratings_val = train_test_split(users,
                movies, ratings, test_size=0.1, random_state=r)

        # build model
        user_input = Input(shape=(1,), dtype='int64')
        movie_input = Input(shape=(1,), dtype='int64')
        user_embedding = Embedding(input_dim=6041, output_dim=d, embeddings_initializer=Orthogonal())(user_input)
        user_embedding = Flatten()(user_embedding)
        movie_embedding = Embedding(input_dim=3953, output_dim=d, embeddings_initializer=Orthogonal())(movie_input)
        movie_embedding = Flatten()(movie_embedding)

        predicted_preference = dot(inputs=[user_embedding, movie_embedding], axes=1)
        predicted_preference = Dense(1, bias_initializer='ones', activation='linear')(predicted_preference)

        model = Model(inputs=[user_input, movie_input], outputs=predicted_preference)
        model.compile(loss='mse', optimizer='rmsprop')

        checkpointer = ModelCheckpoint(filepath='MF_%d_%d.h5' % (d,r),
                                       monitor='val_loss',save_best_only=True,
                                       verbose=1)

        model.fit([users_train, movies_train], ratings_train,
            batch_size=256, epochs=35,
            validation_data=([users_val, movies_val], ratings_val),
            callbacks=[checkpointer],
            verbose=1)


    else:
    
        model = load_model('/home/mhfu/ML2017FALL/hw5/MF_%d_%d.h5' % (d,r))
        #test_data_path = '/home/mhfu/ML2017FALL/hw5/test.csv'
        test_data_path = sys.argv[1]
        with open(test_data_path, 'r') as f:
            f.readline()
            test_user = []
            test_movie = []
            for line in f:
                line = line.strip('\n').split(',')
                test_user.append(int(line[1]))
                test_movie.append(int(line[2]))
                
        test_user = np.array(test_user)
        test_movie = np.array(test_movie)

        result = np.squeeze(model.predict([test_user, test_movie], verbose=1))

        #with open('/home/mhfu/ML2017FALL/hw5/result_%d_%d.csv' % (d,r),'w') as f:
        with open(sys.argv[2],'w') as f:
            f.write('TestDataID,Rating\n')
            for i, r in enumerate(result):
                # clipping on result
                if r <- 1.:
                    f.write('%d,%.4f\n' %(i+1, 1.))
                elif r >= 5.:
                    f.write('%d,%.4f\n' %(i+1, 5.))
                else:
                    f.write('%d,%.4f\n' %(i+1, r))
