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

def enable_ptvsd(port = None, secret='secret'):
    '''
    Enable PTVSD and wait for attach

    Params:
        port:   int port on which to listen on 
        secret: string  secret
    '''
    import ptvsd
    if (port):
        ptvsd.enable_attach(secret, address=('0.0.0.0', port))
    else:
        ptvsd.enable_attach(secret='secret')
    ptvsd.wait_for_attach()