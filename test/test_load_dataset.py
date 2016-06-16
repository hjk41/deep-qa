#!/usr/bin/python

import argparse
import logging
import os
import sys
import time

sys.path.append('../util')
import datasets
import embedding
import utils

if (__name__ == '__main__'):
    '''
    Test the performance of load embedding and load dataset
    '''
    utils.setup_logger()

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

    # test load embedding speed
    logging.info('Start loading embedding.')
    t1 = time.time()
    alphabet, word_vectors = embedding.load_embedding(args.embedding)
    t2 = time.time()
    embedding_size = os.path.getsize(args.embedding)
    logging.info('Completed loading embedding at {} MB/s'
                 .format(embedding_size/1024/1024/(t2-t1)))
    # test load dataset speed
    logging.info('Start loading dataset.')
    t1 = time.time()
    dataset = datasets.load_dataset(args.input, alphabet)
    t2 = time.time()
    dataset_size = os.path.getsize(os.path.join(args.input, 'qids.pickle'))
    dataset_size += os.path.getsize(os.path.join(args.input, 'questions.pickle'))
    dataset_size += os.path.getsize(os.path.join(args.input, 'answers.pickle'))
    dataset_size += os.path.getsize(os.path.join(args.input, 'labels.pickle'))
    logging.info('Completed loading dataset at {} MB/s'
                 .format(dataset_size/1024/1024/(t2-t1)))