import re
import os
import numpy as np
import cPickle
import subprocess
import sys
import string
import shutil
import argparse
from collections import defaultdict
from utils import load_bin_vec

from alphabet import Alphabet
import ptvsd
#ptvsd.enable_attach(secret='secret')
#ptvsd.wait_for_attach()

max_sent_size = 80
'''
UNKNOW_WORD_IDX = len(vocab)
EMPTY_WORD_IDX = len(vocab + 1)
'''

def load_xml(fname, skip_long_sent):
  '''
  load a sample file stored as xml
  '''
  print('loading file {}'.format(fname))
  lines = open(fname).readlines()
  qids, questions, answers, labels = [], [], [], []
  num_skipped = 0
  prev = ''
  question2qid = {}
  curr_qid = 0
  for i, line in enumerate(lines):
    line = line.strip()
    if prev and prev.startswith('<question>'):
      question = line.lower()
      if not question in question2qid:
        qid = curr_qid
        question2qid[question] = curr_qid
        curr_qid += 1
      else:
        qid = question2qid[question]
      question = question.split('\t')

    label = re.match('^<(positive|negative)>', prev)
    prev = line
    if label:
      label = label.group(1)
      label = 1.0 if label == 'positive' else 0.0
      answer = line.lower().split('\t')
      if len(answer) > max_sent_size:
        if (skip_long_sent == True):
          num_skipped += 1
          continue
        else:
          #print('\n{}: {}\n{}|||{}'.format(label, ' '.join(question), ' '.join(answer[:max_sent_size]), ' '.join(answer[max_sent_size:])))
          answer = answer[:max_sent_size]
      labels.append(label)
      answers.append(answer)
      questions.append(question)
      qids.append(qid)
  # print sorted(qid2num_answers.items(), key=lambda x: float(x[0]))
  print 'num_skipped: ', num_skipped
  return question2qid.keys(), qids, questions, answers, labels, []

def passage2list(psg):
  '''
  split the passage into a list of words
  punctuations will be split from words if there is no whitespace between a puctuation and a word
  '''
  list = []
  pos = 0
  wordStart = pos
  isFirstChar = True
  lastIsPunc = False
  n = len(psg)
  while (pos < n):
    c = psg[pos]
    if c == ' ':
      if (not isFirstChar):
        list.append(psg[wordStart:pos])
        lastIsPunc = False
        isFirstChar = True
    elif (c in "!\"#$%&'()*+,./:;<=>?@[\]^`{|}~"):
      if isFirstChar:
        wordStart = pos
      elif (not isFirstChar) and (not lastIsPunc):
        # punctuation following a word, seperate it
        list.append(psg[wordStart:pos])
        wordStart = pos
      lastIsPunc = True
      isFirstChar = False
    else:
      if isFirstChar:
        wordStart = pos
      elif (not isFirstChar) and lastIsPunc:
				# word following a punctuation, seperate it
        list.append(psg[wordStart:pos])
        wordStart = pos
      lastIsPunc = False
      isFirstChar = False
    pos += 1
  if not isFirstChar:
    list.append(psg[wordStart:pos])
  return list    

def load_tsv(fname, skip_long_sent=False, treat_good='remove'):
  # skip tsv header
  #header = lines.pop(0)
  #print 'fields: ', header
  assert treat_good in ['remove', 'positive', 'negative']
  qids, questions, answers, labels = [], [], [], []
  lineids = []
  curr_qid = 0
  num_skipped = 0
  question2qid = {}
  for i, line in enumerate(open(fname)):
    line = line.strip().lower()
    # Query   Url     PassageID       Passage Rating1 Rating2
    qupprr=line.split('\t')
    if len(qupprr) != 6:
      print('error parsing line', i)
      print('line:\n', line)
      exit(1)
    q = qupprr[0].lower()
    if not q in question2qid:
      question2qid[q] = curr_qid
      qid = curr_qid
      curr_qid += 1
    else:
      qid = question2qid[q]
    question = passage2list(q)      ### should we convert to lower case?
    answer = passage2list(qupprr[3].lower())
    if len(answer) > max_sent_size:
      if(skip_long_sent):
        num_skipped += 1
        continue
      else:
        answer = answer[:max_sent_size]
    r2 = qupprr[5].lower()
    if r2 == 'perfect':
      label = 1.0
    elif r2 == 'good':
      if (treat_good == 'remove'):
        continue
      elif (treat_good == 'positive'):
        label = 1.0
      else: # negative
        label = 0.0
    else:
      label = 0.0
    labels.append(label)
    answers.append(answer)
    questions.append(question)
    qids.append(qid)
    lineids.append(i)
  return question2qid.keys(), qids, questions, answers, labels, lineids

def load_data(fname, skip_long_sent = True, resample = True, treat_good = 'remove'):
  basename = os.path.basename(fname)
  name, ext = os.path.splitext(basename)
  if (ext == '.tsv'):
    return load_tsv(fname, skip_long_sent, treat_good)
  else:
    return load_xml(fname, skip_long_sent)  

def compute_overlap_features(questions, answers, word2df=None, stoplist=None):
  '''
  compute overlap features
  there are two overlap features: overlap ratio with and without IDF
  '''
  word2df = word2df if word2df else {}
  stoplist = stoplist if stoplist else set()
  feats_overlap = []
  for question, answer in zip(questions, answers):
    q_set = set([q for q in question if q not in stoplist])
    a_set = set([a for a in answer if a not in stoplist])
    word_overlap = q_set.intersection(a_set)
    # overlap = num_overlap_words / (words_in_q * words_in_a)
    overlap = float(len(word_overlap)) / (len(q_set) * len(a_set) + 1e-8)
    # overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))
    df_overlap = 0.0
    for w in word_overlap:        ### should we count the word frequency?
      df_overlap += word2df[w]
    #total_dfs = 0.0
    #for w in q_set:
    #  total_dfs += word2df[w]
    #for w in a_set:
    #  total_dfs += word2df[w]
    # df_overlap = total_overlap_IDF / (words_in_q + words_in_a)
    df_overlap /= (len(q_set) + len(a_set))
    #df_overlap /= total_dfs

    feats_overlap.append(np.array([
                         overlap,
                         df_overlap,
                         ]))
  return np.array(feats_overlap)


def compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length):
  '''
  compute overlap feature of q and a
  for each pair of q and a, output are two int32 arrays corresponding to q and a
  in which array[i]==1 when the word is a overlap word in both q and a
  '''
  stoplist = stoplist if stoplist else []
  feats_overlap = []
  q_indices, a_indices = [], []
  for question, answer in zip(questions, answers):
    q_set = set([q for q in question if q not in stoplist])
    a_set = set([a for a in answer if a not in stoplist])
    word_overlap = q_set.intersection(a_set)

    # why *2? it actually makes all the elements 2. 
    # so 0 is non-overlap, 1 is overlap, and 2 is empty word
    q_idx = np.ones(q_max_sent_length) * 2    
    for i, q in enumerate(question):
      value = 0
      if q in word_overlap:
        value = 1
      q_idx[i] = value
    q_indices.append(q_idx)

    a_idx = np.ones(a_max_sent_length) * 2
    for i, a in enumerate(answer):
      value = 0
      if a in word_overlap:
        value = 1
      a_idx[i] = value
    a_indices.append(a_idx)

  q_indices = np.vstack(q_indices).astype('int32')
  a_indices = np.vstack(a_indices).astype('int32')

  return q_indices, a_indices

#def compute_dfs(docs):
#  word2df = defaultdict(float)
#  for doc in docs:
#    for w in set(doc):
#      word2df[w] += 1.0
#  num_docs = len(docs)
#  for w, value in word2df.iteritems():
#    word2df[w] /= np.math.log(num_docs / value)   # why /=? shouldn't it be =?
#  return word2df

def compute_dfs(docs):
  word2df = defaultdict(float)
  for doc in docs:
    for w in set(doc):
      word2df[w] += 1.0
  num_docs = len(docs)
  for w, value in word2df.iteritems():
    word2df[w] = np.math.log(num_docs / value)   # why /=? shouldn't it be =?
  return word2df

def add_to_vocab(data, alphabet):
  for sentence in data:
    for token in sentence:
      alphabet.add(token)


def convert2indices(data, alphabet, unknown_word_idx, empty_word_idx, max_sent_length=40):
  data_idx = []
  unknown_words = set()
  for sentence in data:
    ex = np.ones(max_sent_length) * empty_word_idx
    for i, token in enumerate(sentence):
      idx = alphabet.get(token, unknown_word_idx)
      if (idx == unknown_word_idx):
          unknown_words.add(token)
      ex[i] = idx
    data_idx.append(ex)
  data_idx = np.array(data_idx, dtype='int32')
  return (data_idx, unknown_words)

def calculate_tfidf(data, word2dfs, max_sent_length):
  data_tfidf = []
  for sentence in data:
    ex = np.zeros(max_sent_length)
    for i, token in enumerate(sentence):
      tfidf = word2dfs.get(token)
      ex[i] = tfidf
    data_tfidf.append(ex)
  data_tfidf = np.array(data_tfidf).astype('float32')
  return data_tfidf

def convert_dataset(qids, questions, answers, labels, lineids,
    stoplist, 
    word2dfs,
    word2id,
    unknown_word_idx,
    prefix):
  '''
  convert a dataset into the feature files we need, and store
  the result in outdir
  '''
  # summarize max sentense length
  q_max_sent_length = max(map(lambda x: len(x), questions))
  a_max_sent_length = max(map(lambda x: len(x), answers))
  print 'q_max_sent_length', q_max_sent_length
  print 'a_max_sent_length', a_max_sent_length

  overlap_feats = compute_overlap_features(questions, answers, stoplist=None, word2df=word2dfs)
  overlap_feats_stoplist = compute_overlap_features(questions, answers, stoplist=stoplist, word2df=word2dfs)
  overlap_feats = np.hstack([overlap_feats, overlap_feats_stoplist])

  qids = np.array(qids)
  labels = np.array(labels).astype('float32')
  _, counts = np.unique(labels, return_counts=True)
  print "label frequencies: ", counts / float(np.sum(counts))
  print "samples: ", len(labels)

  q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length)
  questions_idx, q_unknown_words = convert2indices(questions, word2id, unknown_word_idx, unknown_word_idx + 1, q_max_sent_length)
  answers_idx, a_unknown_words = convert2indices(answers, word2id, unknown_word_idx, unknown_word_idx + 1, a_max_sent_length)
  unknown_words = q_unknown_words.union(a_unknown_words)
  q_tfidf = calculate_tfidf(questions, word2dfs, q_max_sent_length)
  a_tfidf = calculate_tfidf(answers, word2dfs, a_max_sent_length)

  print('dumping files')
  # question ids for each sample
  np.save(os.path.join(prefix, 'qids.npy'), qids)
  # questions of each sample, represented by word indices
  np.save(os.path.join(prefix, 'questions.npy'), questions_idx)
  # answers of each sample, represented by word indices
  np.save(os.path.join(prefix, 'answers.npy'), answers_idx)
  # labels of each sample, represented as float32
  np.save(os.path.join(prefix, 'labels.npy'), labels)
  # overlap features, including features with and without stoplist
  np.save(os.path.join(prefix, 'overlap_feats.npy'), overlap_feats)
  np.save(os.path.join(prefix, 'q_overlap_indices.npy'), q_overlap_indices)
  np.save(os.path.join(prefix, 'a_overlap_indices.npy'), a_overlap_indices)
  np.save(os.path.join(prefix, 'q_tfidf.npy'), q_tfidf)
  np.save(os.path.join(prefix, 'a_tfidf.npy'), a_tfidf)
  np.save(os.path.join(prefix, 'lineids.npy'), np.array(lineids))
  with open(os.path.join(prefix, 'nonembed.txt'), 'w') as f:
    for w in unknown_words:
      f.write('{}\n'.format(w))

def load_glove(embeddingfile, words):
  word2vec = {}
  with open(embeddingfile) as f:
    i = 0
    for line in f:
      i += 1
      if (i % 1000 == 0):
        sys.stdout.write('load {} lines\t\t\r'.format(i))
        sys.stdout.flush()
      fields = line.split()
      w = fields[0]
      if (w not in words):
        continue
      word2vec[w] = np.array([float(f) for f in fields[1:]])
  sys.stdout.write('\n')
  return word2vec

def dump_embedding(outdir, embeddingfile, alphabet):
  words = alphabet.keys()
  print "Vocab size: ", len(alphabet)
  #word2vec = load_bin_vec(embeddingfile, words)
  word2vec = load_glove(embeddingfile, words)
  print "embedded vocab: ", len(word2vec)
  ndim = len(word2vec[word2vec.keys()[0]])
  print 'embedding dim: ', ndim
  random_words_count = 0
  np.random.seed(321)
  vocab_emb = np.zeros((len(alphabet) + 1, ndim))
  dummy_word_emb = np.random.uniform(-0.25, 0.25, ndim)
  print('out-of-vocab words:')
  for word, idx in alphabet.iteritems():
    word_vec = word2vec.get(word, None)
    if word_vec is None:
      print(word)
      word_vec = np.random.uniform(-0.25, 0.25, ndim)
      #word_vec = dummy_word_emb
      #word_vec = np.zeros(ndim)
      #word_vec = np.ones(ndim)
      random_words_count += 1
    vocab_emb[idx] = word_vec
  print "Using zero vector as random"
  print 'random_words_count', random_words_count
  print 'vocab_emb.shape', vocab_emb.shape
  outfile = os.path.join(outdir, 'emb.npy'.format(os.path.basename(embeddingfile)))
  print 'saving embedding file', outfile
  np.save(outfile, vocab_emb)

def sample(list, idx):
  return [list[i] for i in idx]

if __name__ == '__main__':
  '''
  parses a dataset into features

  The input can be xml format or tsv format
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', help='input file', required=True)
  parser.add_argument('-o', '--output', help='output dir', required=True)
  parser.add_argument('-e', '--embed', help='embedding dir', required=True)
  parser.add_argument('-r', '--resample', type=bool, default=False, help='whether to do resampling, 1/0')
  parser.add_argument('-g', '--good', type=str, default='remove', choices=['remove', 'positive', 'negative'], help='how to treat good examples, remove/positive/negative')
  args = parser.parse_args()
  
  inputfile = args.input
  outputdir = args.output
  embeddingdir = args.embed
  resample = args.resample
  treat_good = args.good


  # compose output file names
  print("\n================================\n"
      "using:\n"
      "    inputfile={}\n"
      "will output:\n"
      "    {}/(qids, questions, answers, labels, overlap).npy\n"
      "words with no embedding will be stored in {}/nonembed.txt\n"
      "================================".format(inputfile, outputdir, outputdir))

  if not os.path.exists(outputdir):
    os.makedirs(outputdir)

  # load stoplist
  stoplist = set()
  import string
  punct = set(string.punctuation)
  #stoplist.update(punct)

  # compute word frequencies
  print('loading input file {}'.format(inputfile))
  unique_questions, qids, questions, answers, labels, lineids = load_data(inputfile, skip_long_sent=False, resample = resample, treat_good=treat_good)
  docs = answers + unique_questions
  word2dfs = compute_dfs(docs)

  # load embedding
  print('loading embedding from {}'.format(embeddingdir))
  word2id = cPickle.load(open(os.path.join(embeddingdir, 'vocab.pickle')))
  unknown_word_id = len(word2id)

  # Convert datasets
  convert_dataset(qids = qids, 
      questions = questions,
      answers = answers,
      labels = labels,
      lineids = lineids,
      stoplist = stoplist,
      word2dfs = word2dfs,
      word2id = word2id,
      unknown_word_idx = unknown_word_id,
      prefix = outputdir)