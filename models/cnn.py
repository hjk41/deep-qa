#!/usr/bin/python

import keras.backend as K
from keras.layers import Merge, Convolution1D, MaxPooling1D
from keras.models import Model

def qa_cnn(q_vectors, a_vectors,
           num_conv_kernels,
           conv_kernel_size):
    '''
    This model simply stacks N layers of convolution and pooling, and then
    use a max pooling layer so the output of Q and A have fixed size. Finally,
    a fully connected layer is used and then a logistic regression used to
    output a similarity score for Q and A.

    Params:
        q_vectors:    Tensor(n_samples, sentense_length, dim)
        a_vectors:    Tensor(n_samples, sentense_length, dim)
        num_conv_kernels:    int[]    number of conv kernels for each conv layer
        conv_kernel_size:    int[]    kernel size for each layer

    Returns:
        Tensor(n_samples) scores for each sample
    '''
    assert len(num_conv_kernels) == len(conv_kernel_size)
    for n_kernels, kernel_size in zip(num_conv_kernels, conv_kernel_size):
        conv = Convolution1D(n_kernels, kernel_size, 
                             activation='tanh', border_mode='same')

    pass