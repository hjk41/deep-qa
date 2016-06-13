#!/usr/bin/python
'''
This is utility file, containing functions to work with data sets and embeddings.
'''

import argparse
import logging
import numpy
import os
import pickle

from utils import setup_logger, enable_ptvsd

def load_embedding_w2v(fname):
    """
    Loads word vecs from Google (Mikolov) word2vec
  
    Params:
        fname: string   path of the embedding file

    Returns:
        dictionary of {string:numpy.array}
    """
    logging.info('Loading embedding file {}'.format(fname))
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, embedding_dim = map(int, header.split())
        binary_len = numpy.dtype('float32').itemsize * embedding_dim
        logging.info('vocab_size={}, embedding_dim={}'.format(vocab_size, embedding_dim))
        for i, line in enumerate(range(vocab_size)):
            if (i % 100000 == 0):
                logging.info('read {} lines'.format(i))
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word).decode('utf-8')
                    break
                if ch != b'\n':
                    word.append(ch)
            word_vecs[word] = numpy.fromstring(f.read(binary_len), dtype='float32')
    logging.info('Loading complete.')
    return word_vecs

def load_embedding_glove(fname):
    """
    Loads word vecs from Glove
  
    Params:
        fname: string   path of the embedding file

    Returns:
        dictionary of {string:numpy.array}
    """
    logging.info('Loading embedding file {}'.format(fname))
    word2vec = {}
    for line in open(fname):
        fields = line.split()
        w = fields[0]
        word2vec[w] = numpy.array([float(f) for f in fields[1:]])
    embedding_dim = next(iter(word2vec.values())).size
    logging.info('vocab_size=%d, embedding_dim=%d', len(word2vec), embedding_dim)
    logging.info('Loading complete.')
    return word2vec

def convert_embedding(outputfile, embeddingfile, format='word2vec'):
    """
    Convert embedding file into a format easier to read.
    Embedding file will be loaded and convert into a {string:numpy.array}
    dictionary, and then dumpped into the outputfile with pickle.
  
    Params:
        outputfile: string   path of output file
        embeddingfile:  string   path of the embedding file
        format: string  can be word2vec or glove
    """
    if (format == 'word2vec'):
        word2vec = load_embedding_w2v(embeddingfile)
    elif (format == 'glove'):
        word2vec = load_embedding_glove(embeddingfile)
    else:
        logging.error('embedding format can only be word2vec or glove')

    logging.info('Dumpping embedding to {}'.format(outputfile))
    pickle.dump(word2vec, open(outputfile, 'wb'))
    logging.info('Dumpping complete.')

def load_parsed_embedding(embeddingfile):
    '''
    Load parsed embedding file.

    Params:
        embeddingfile:    string  path of the embedding file generated by
                                  convert_embedding

    Returns:
        dictionary of {string:numpy.array}
    '''
    return pickle.load(open(embeddingfile, 'rb'))

if (__name__ == '__main__'):
    '''
    Parses the embedding file into pickle format so later we can load them easily
    '''
    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input embedding file', 
                        required=True)
    parser.add_argument('-o', '--output', help='output embedding file', 
                        required=True)
    parser.add_argument('-m', '--format', type=str, default='word2vec', 
                        choices=['word2vec', 'glove'], 
                        help='is this file generated by word2vec or glove?')
    parser.add_argument('-d', '--debug', default=False, action='store_true', 
                        help='enable ptvsd debugging')
    args = parser.parse_args()

    if (args.debug):
        enable_ptvsd()
    
    parent_dir = os.path.abspath(os.path.join(args.output, os.pardir))
    if (not os.path.exists(parent_dir)):
        os.makedirs(parent_dir)
    convert_embedding(args.output, args.input, args.format)