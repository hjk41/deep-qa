#!/bin/python

import sys
import numpy as np
import cPickle
import os

def load_glove(filename):
  w2v = {}
  count = 0
  for line in open(filename):
    count += 1
    if (count % 100000 == 0):
      print('.')
    f = line.split()
    w2v[f[0]] = np.array([float(ff) for ff in f[1:]])
  return w2v

def load_w2v(fname):
  """
  Loads word vecs from Google (Mikolov) word2vec
  """
  word_vecs = {}
  with open(fname, "rb") as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    print('total vocab size: {}'.format(vocab_size))
    for i, line in enumerate(xrange(vocab_size)):
      if i % 100000 == 0:
        print('.')
      word = []
      while True:
        ch = f.read(1)
        if ch == ' ':
            word = ''.join(word)
            break
        if ch != '\n':
            word.append(ch)
      word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
    print "done"
    return word_vecs


def main(argv):
  if (len(argv) != 2):
    print('usage: parse2bin.py embeddingfile')
    exit(1)
  fpath = argv[1]
  dir, fname = os.path.split(fpath)
  print('loading embedding file {}'.format(fpath))
  if (fname.startswith('glove')):
    w2v = load_glove(fpath)
  else:
    w2v = load_w2v(fpath)
  out_fname = '.'.join(fname.split('.')[:-1])
  vocab = {}
  vectors = []
  wid = 0
  for w, v in w2v.items():
    vocab[w] = wid
    wid += 1
    vectors.append(v)
  vectors = np.array(vectors)
  vocab_file = os.path.join(dir, out_fname + '.vocab.pickle')
  print('dumping vocab file to: {}'.format(vocab_file))
  cPickle.dump(vocab, open(vocab_file, 'w'))
  vector_file = os.path.join(dir, out_fname + '.npy')
  print('dumping vector file to: {}'.format(vector_file))
  np.save(vector_file, vectors)

if __name__ == '__main__':
  main(sys.argv)
