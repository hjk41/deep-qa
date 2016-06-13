#!/usr/bin/python
'''
This is utility
'''
import logging

def setup_logger(level=logging.INFO):
    '''
    Setup root logger so we can easily use it

    Params:
        level:  string  logging level
    '''
    logging.root.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(name)s-%(levelname)s: %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)