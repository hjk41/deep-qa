#!/usr/bin/python
'''
This file implements some common operations for preparing the models.
'''

import keras.backend as K
from keras.layers import Input, merge, Embedding

def add_overlap_feature(q, a, 
                        q_overlap, a_overlap,
                        overlap_embedding_dim=5):
    '''
    Add overlap feature to q and a

    Params:
        q:    Keras.layers.Input    questions, shape=(num_samples, q_length)
        a:    Keras.layers.Input    answers, shape=(num_samples, a_length)
        q_overlap:    Keras.layers.Input    overlap feature
        a_overlap:    Keras.layerrs.Input    overlap feature
        overlap_embedding_dim:    int    number of embedding dim to use

    Returns:
        q_merged:    concate(q, embed(q_overlap))
        a_merged:    concate(a, embed(a_overlap))
    '''
    embed = Embedding(input_dim=1, output_dim=5)
    q_overlap_embed = embed(q_overlap)
    a_overlap_embed = embed(a_overlap)
    q_merged = merge([q, q_overlap_embed], mode='concat')
    a_merged = merge([a, a_overlap_embed], mode='concat')
    return q_merged, a_merged