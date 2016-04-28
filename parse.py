import re
import os
import numpy as np
import cPickle
import subprocess
import sys
import string
from collections import defaultdict
from utils import load_bin_vec

from alphabet import Alphabet
import ptvsd
#ptvsd.enable_attach(secret='secret', address = ('0.0.0.0',9999))
#ptvsd.wait_for_attach()

UNKNOWN_WORD_IDX = 0

def load_xml(fname):
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
    if label:
      label = label.group(1)
      label = 1.0 if label == 'positive' else 0.0
      answer = line.lower().split('\t')
      #if len(answer) > 60:
      if len(answer) > 70:
        #num_skipped += 1
        #continue
        answer = answer[:70]
      labels.append(label)
      answers.append(answer)
      questions.append(question)
      qids.append(qid)
    prev = line
  # print sorted(qid2num_answers.items(), key=lambda x: float(x[0]))
  # print 'num_skipped: ', num_skipped
  return question2qid.keys(), qids, questions, answers, labels

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

def load_tsv(fname):
  lines = open(fname).readlines()
  # skip tsv header
  #header = lines.pop(0)
  #print 'fields: ', header
  qids, questions, answers, labels = [], [], [], []
  curr_qid = 0
  num_skipped = 0
  question2qid = {}
  for i, line in enumerate(lines):
    line = line.strip().lower()
    # Query   Url     PassageID       Passage Rating1 Rating2
    qupprr=line.split('\t')
    if len(qupprr) != 6:
      print('error parsing line', i)
      print('line:\n', line)
      exit(1)
    q = qupprr[0]
    if not q in question2qid:
      question2qid[q] = curr_qid
      qid = curr_qid
      curr_qid += 1
    else:
      qid = question2qid[q]
    question = passage2list(q)      ### should we convert to lower case?
    answer = passage2list(qupprr[3])
    if len(answer) > 70:
      answer = answer[:70]
    r2 = qupprr[5].lower()
    if r2 == 'perfect':
      label = 1.0
    elif r2 == 'good':
      label = 1.0
    else:
      label = 0.0
    labels.append(label)
    answers.append(answer)
    questions.append(question)
    qids.append(qid)
    prev = line
  return question2qid.keys(), qids, questions, answers, labels

def load_data(fname):
  basename = os.path.basename(fname)
  name, ext = os.path.splitext(basename)
  if (ext == '.tsv'):
    return load_tsv(fname)
  else:
    return load_xml(fname)  

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


def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=40):
  data_idx = []
  for sentence in data:
    ex = np.ones(max_sent_length) * dummy_word_idx
    for i, token in enumerate(sentence):
      idx = alphabet.get(token, UNKNOWN_WORD_IDX)
      ex[i] = idx
    data_idx.append(ex)
  data_idx = np.array(data_idx).astype('int32')
  return data_idx

def convert_dataset(qids, questions, answers, labels, 
    stoplist, 
    word2dfs,
    alphabet,
    dummy_word_idx,
    q_max_sent_length,
    a_max_sent_length,
    outdir, basename):
  '''
  convert a dataset into the feature files we need, and store
  the result in outdir
  '''

  overlap_feats = compute_overlap_features(questions, answers, stoplist=None, word2df=word2dfs)
  overlap_feats_stoplist = compute_overlap_features(questions, answers, stoplist=stoplist, word2df=word2dfs)
  overlap_feats = np.hstack([overlap_feats, overlap_feats_stoplist])
  print 'overlap_feats shape=', overlap_feats.shape

  qids = np.array(qids)
  labels = np.array(labels).astype('float32')
  _, counts = np.unique(labels, return_counts=True)
  print "label frequencies: ", counts / float(np.sum(counts))
  print "unique questions: ", len(np.unique(qids))
  print "samples: ", len(labels)

  q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length)

  questions_idx = convert2indices(questions, alphabet, dummy_word_idx, q_max_sent_length)
  answers_idx = convert2indices(answers, alphabet, dummy_word_idx, a_max_sent_length)
  print 'answers_idx', answers_idx.shape

  # question ids for each sample
  np.save(os.path.join(outdir, '{}.qids.npy'.format(basename)), qids)
  # questions of each sample, represented by word indices
  np.save(os.path.join(outdir, '{}.questions.npy'.format(basename)), questions_idx)
  # answers of each sample, represented by word indices
  np.save(os.path.join(outdir, '{}.answers.npy'.format(basename)), answers_idx)
  # labels of each sample, represented as float32
  np.save(os.path.join(outdir, '{}.labels.npy'.format(basename)), labels)
  # overlap features, including features with and without stoplist
  np.save(os.path.join(outdir, '{}.overlap_feats.npy'.format(basename)), overlap_feats)
  np.save(os.path.join(outdir, '{}.q_overlap_indices.npy'.format(basename)), q_overlap_indices)
  np.save(os.path.join(outdir, '{}.a_overlap_indices.npy'.format(basename)), a_overlap_indices)

def dump_embedding(outdir, embeddingfile, alphabet):
  words = alphabet.keys()
  print "Vocab size: ", len(alphabet)
  word2vec = load_bin_vec(embeddingfile, words)
  ndim = len(word2vec[word2vec.keys()[0]])
  print 'embedding dim: ', ndim
  random_words_count = 0
  np.random.seed(321)
  vocab_emb = np.zeros((len(alphabet) + 1, ndim))
  for word, idx in alphabet.iteritems():
    word_vec = word2vec.get(word, None)
    if word_vec is None:
      word_vec = np.random.uniform(-0.25, 0.25, ndim)
      random_words_count += 1
    vocab_emb[idx] = word_vec
  print "Using zero vector as random"
  print 'random_words_count', random_words_count
  print 'vocab_emb.shape', vocab_emb.shape
  outfile = os.path.join(outdir, 'emb_{}.npy'.format(os.path.basename(embeddingfile)))
  print 'saving embedding file', outfile
  np.save(outfile, vocab_emb)

def sample(list, idx):
  return [list[i] for i in idx]

if __name__ == '__main__':
  '''
  parses a dataset (including train, validation, and test) into float features

  The input can be xml format or tsv format
  If validation file is not given, it takes 1/6 of randomly sampled samples
  from training set
  '''
  if (len(sys.argv) < 4):
    print("usage: parse.py outputdir trainfile testfile [validationfile]")
    exit(1)
  
  # parse command line arguments
  outdir = sys.argv[1]
  train = sys.argv[2]
  test = sys.argv[3]
  dev = "" if len(sys.argv) < 5 else sys.argv[4]
  print("using:\n"
      "    outputdir={}\n"
      "    train={}\n"
      "    validation={}\n"
      "    test={}".format(outdir, train, dev, test))

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  # load stoplist
  stoplist = set()
  import string
  punct = set(string.punctuation)
  #stoplist.update(punct)

  # merge inputs to compute word frequencies
  _, ext = os.path.splitext(os.path.basename(train))
  all_fname = "/tmp/trec-merged" + ext
  files = ' '.join([train, dev, test])
  subprocess.call("/bin/cat {} > {}".format(files, all_fname), shell=True)
  unique_questions, qids, questions, answers, labels = load_data(all_fname)

  docs = answers + unique_questions
  word2dfs = compute_dfs(docs)
  print word2dfs.items()[:10]

  # map words to ids
  alphabet = Alphabet(start_feature_id=0)
  alphabet.add('UNKNOWN_WORD_IDX')
  add_to_vocab(answers, alphabet)
  add_to_vocab(questions, alphabet)
  basename = os.path.basename(train)
  cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab.pickle'), 'w'))
  print "alphabet size=", len(alphabet)

  # dump embedding file
  dummy_word_idx = alphabet.fid
  dump_embedding(outdir, 'embeddings/aquaint+wiki.txt.gz.ndim=50.bin', alphabet)

  # summarize max sentense length
  q_max_sent_length = max(map(lambda x: len(x), questions))
  a_max_sent_length = max(map(lambda x: len(x), answers))
  print 'q_max_sent_length', q_max_sent_length
  print 'a_max_sent_length', a_max_sent_length

  # Convert datasets
  train_unique_qs, train_qids, train_questions, train_answers, train_labels = load_data(train)
  test_unique_qs, test_qids, test_questions, test_answers, test_labels = load_data(test)
  if (dev == ""):
    # get 1/6 of train data and put it in dev
    train_size = len(train_qids)
    sample_idx = np.arange(train_size)
    np.random.shuffle(sample_idx)
    dev_size = train_size / 6;
    dev_samples = sample_idx[:dev_size]
    train_samples = sample_idx[dev_size:]
    dev_qids = sample(train_qids,dev_samples)
    train_qids = sample(train_qids,train_samples)
    dev_questions = sample(train_questions,dev_samples)
    train_questions = sample(train_questions,train_samples)
    dev_answers = sample(train_answers,dev_samples)
    train_answers = sample(train_answers,train_samples)
    dev_labels = sample(train_labels,dev_samples)
    train_labels = sample(train_labels,train_samples)
  else:
    dev_unique_qs, dev_qids, dev_questions, dev_answers, dev_labels = load_data(dev)
  convert_dataset(qids = train_qids, 
      questions = train_questions,
      answers = train_answers,
      labels = train_labels,
      stoplist = stoplist,
      word2dfs = word2dfs,
      alphabet = alphabet,
      dummy_word_idx = dummy_word_idx,
      q_max_sent_length = q_max_sent_length,
      a_max_sent_length = a_max_sent_length,
      outdir = outdir,
      basename = "train")
  convert_dataset(qids = dev_qids, 
      questions = dev_questions,
      answers = dev_answers,
      labels = dev_labels,
      stoplist = stoplist,
      word2dfs = word2dfs,
      alphabet = alphabet,
      dummy_word_idx = dummy_word_idx,
      q_max_sent_length = q_max_sent_length,
      a_max_sent_length = a_max_sent_length,
      outdir = outdir,
      basename = "dev")
  convert_dataset(qids = test_qids, 
      questions = test_questions,
      answers = test_answers,
      labels = test_labels,
      stoplist = stoplist,
      word2dfs = word2dfs,
      alphabet = alphabet,
      dummy_word_idx = dummy_word_idx,
      q_max_sent_length = q_max_sent_length,
      a_max_sent_length = a_max_sent_length,
      outdir = outdir,
      basename = "test")

  

