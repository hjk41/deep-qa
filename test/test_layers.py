#!/usr/bin/python
import argparse
import sys

import keras.backend as K
from keras.models import Model
from keras.layers import Input

sys.path.append('../util')
from utils import setup_logger, enable_ptvsd
import embedding
import datasets
import layers

if __name__ == '__main__':
    '''
    Test different layers
    '''
    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input dataset directory', 
                        required=True)
    parser.add_argument('-e', '--embedding', help='embedding pickle file', 
                        required=True)
    parser.add_argument('-d', '--debug', default=False, action='store_true', 
                        help='enable ptvsd debugging')
    args = parser.parse_args()

    if (args.debug):
        enable_ptvsd()

    alphabet, word_vectors = embedding.load_embedding(args.embedding)
    dataset = datasets.load_dataset(args.input,
                                    alphabet, True, None)
    q_vectors = datasets.extract_word_vectors(dataset.q, word_vectors)
    a_vectors = datasets.extract_word_vectors(dataset.a, word_vectors)
    q_length = dataset.q.shape[1]
    a_length = dataset.a.shape[1]
    embedding_dim = word_vectors.shape[1]
    overlap_embedding_dim = 5

    Q = Input(shape=(q_length, embedding_dim))
    A = Input(shape=(a_length, embedding_dim))
    QO = Input(shape=(q_length, ), dtype='int32')
    AO = Input(shape=(a_length, ), dtype='int32')
    O = layers.add_overlap_feature(Q, A, QO, AO, 5)
    model = Model(input=[Q,A,QO,AO], output=O)
    model.compile(optimizer='adagrad', loss='mse')
    output = model.predict([q_vectors, a_vectors, dataset.q_overlap, dataset.a_overlap])
    print(output)