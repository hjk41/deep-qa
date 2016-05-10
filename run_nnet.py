from datetime import datetime
from sklearn import metrics
from theano import tensor as T
import cPickle
import numpy
import os
import sys
import theano
import time
from collections import defaultdict
import subprocess
import pandas as pd
from tqdm import tqdm

import nn_layers
import sgd_trainer

import warnings
warnings.filterwarnings("ignore")  # TODO remove

import ptvsd
#ptvsd.enable_attach(secret='secret')
#ptvsd.wait_for_attach()

### THEANO DEBUG FLAGS
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = 'high'

n_epochs = 100
batch_size = 50
n_dev_batch = 200
n_iter_per_val = 100
n_outs = 2
learning_rate = 0.1
max_norm = 0
early_stop_epochs = 5
regularize = False
pairwise_feature = False

def conv_layer(batch_size,
    max_sent_length,
    embedding,                # embedding dictionary, from id to vector, static
    embedding_overlap,        # embedding matrix for the overlap feature, will be updated
    numpy_randg,              # numpy random number generator
    x,                        # intput matrix, each sentense is a row, each element is index
    x_overlap,                # overlap matrix
    filter_widths = [5],      # list of filter widths to use, currently [5]
    n_conv_kern = 100,        # number of convolution kernels, currently 100
    n_input_channel = 1,
    pooling_k_max = 1):              # output of max polling, currently 1
  '''
  Build the convolution layer that convs the q and a
  
  Returns a network with input (x, x_overlap)
  '''
  # perform lookup and then pad the matrix so we can later use it in conv
  lookup_table_words = nn_layers.LookupTableFastStatic(W=embedding, pad=max(filter_widths)-1)
  #lookup_table_words = nn_layers.LookupTableFast(W=embedding, pad=max(filter_widths)-1)
  lookup_table_overlap = nn_layers.LookupTableFast(W=embedding_overlap, pad=max(filter_widths)-1)
  # concatenate the outputs of words and overlap into one
  # now the output dim is embedding+embedding_overlap, currently 50+5
  lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap])
  ndim = embedding.shape[1] + embedding_overlap.shape[1]
  input_shape = (batch_size, n_input_channel, max_sent_length + 2*(max(filter_widths)-1), ndim)
  conv_layers = []
  for filter_width in filter_widths:
    filter_shape = (n_conv_kern, n_input_channel, filter_width, ndim)
    conv = nn_layers.Conv2dLayer(rng=numpy_randg, filter_shape=filter_shape, input_shape=input_shape)
    non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0], activation=T.tanh)
    pooling = nn_layers.KMaxPoolLayer(k_max=pooling_k_max)
    conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[conv, non_linearity, pooling])
    conv_layers.append(conv2dNonLinearMaxPool)
  join_layer = nn_layers.ParallelLayer(layers=conv_layers)
  flatten_layer = nn_layers.FlattenLayer()
  nnet = nn_layers.FeedForwardNet(layers=[
                                  lookup_table,
                                  join_layer,
                                  flatten_layer,
                                  ])
  nnet.set_input((T.cast(x, 'int32'), T.cast(x_overlap, 'int32')))
  return nnet

def deep_qa_net(batch_size,
    embedding,                # embedding dictionary, from id to vector, static
    embedding_overlap,        # embedding matrix for the overlap feature, will be updated
    q_max_sent_length,
    a_max_sent_length,
    numpy_randg,              # numpy random number generator
    x_q,                      # input symbols
    x_q_overlap,
    x_a,
    x_a_overlap,
    n_conv_kern = 100,        # number of convolution kernels, currently 100
    n_input_channel = 1,
    q_k_max = 1,              # output of max polling, currently 1
    a_k_max = 1,              
    dropout = 0.5):    
  '''
  Builds deepQA network

  batch_size:               int, size of the batch
  max_sent_length:          int, maximum length of sentense
  embedding:                numpy array, embedding dictionary, from id to vector, static
  embedding_overlap:        numpy array, embedding matrix for the overlap feature, will be updated
  n_conv_kern               int, number of convolution kernels, currently 100
  n_input_channel           int, number of input channel for convolution, currently 1
  '''
  q_filter_widths = [5]
  a_filter_widths = [5]
  ## question conv
  nnet_q = conv_layer(batch_size, q_max_sent_length, 
                      embedding, embedding_overlap, numpy_randg, 
                      x_q, x_q_overlap,
                      q_filter_widths, 100, 1, 1)
  ## answer conv
  nnet_a = conv_layer(batch_size, a_max_sent_length, 
                      embedding, embedding_overlap, numpy_randg, 
                      x_a, x_a_overlap,
                      a_filter_widths, 100, 1, 1)
  q_logistic_n_in = n_conv_kern * len(q_filter_widths) * q_k_max
  a_logistic_n_in = n_conv_kern * len(a_filter_widths) * a_k_max
  if (pairwise_feature):
    ## calculate similarity as sim = q.T * M * a
    # output is (conv_out_a, conv_out_q, similarity)
    pairwise_layer = nn_layers.PairwiseNoFeatsLayer(q_in=q_logistic_n_in,
                                                  a_in=a_logistic_n_in)
    pairwise_layer.set_input((nnet_q.output, nnet_a.output))
    # hidden layer
    # input is (conv_out_a, conv_out_q, similarity)
    n_in = q_logistic_n_in + a_logistic_n_in + 1
    hidden_layer = nn_layers.LinearLayer(numpy_randg, n_in=n_in, n_out=n_in, activation=T.tanh)
    hidden_layer.set_input(pairwise_layer.output)
    hidden_layer2 = nn_layers.LinearLayer(numpy_randg, n_in=n_in, n_out=n_in, activation=T.tanh)
    hidden_layer2.set_input(hidden_layer.output)
    classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)
    classifier.set_input(hidden_layer2.output)
    nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, pairwise_layer, hidden_layer, hidden_layer2, classifier],
                                          name="nnet")
  else:
    n_in = q_logistic_n_in + a_logistic_n_in
    hidden_layer = nn_layers.LinearLayer(numpy_randg, n_in=n_in, n_out=n_in, activation=T.tanh)
    hidden_layer.set_input(T.concatenate([nnet_q.output, nnet_a.output], axis=1))
    hidden_layer2 = nn_layers.LinearLayer(numpy_randg, n_in=n_in, n_out=n_in, activation=T.tanh)
    hidden_layer2.set_input(hidden_layer.output)
    classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)
    classifier.set_input(hidden_layer2.output)
    nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, hidden_layer, hidden_layer2, classifier],
                                          name="nnet")
  return nnet

def load_data(input_dir, prefix):
  q = numpy.load(os.path.join(input_dir, prefix + '.questions.npy')).astype(numpy.float32)
  a = numpy.load(os.path.join(input_dir, prefix + '.answers.npy')).astype(numpy.float32)
  q_overlap = numpy.load(os.path.join(input_dir, prefix + '.q_overlap_indices.npy')).astype(numpy.float32)
  a_overlap = numpy.load(os.path.join(input_dir, prefix + '.a_overlap_indices.npy')).astype(numpy.float32)
  y = numpy.load(os.path.join(input_dir, prefix + '.labels.npy')).astype(numpy.float32)
  qids = numpy.load(os.path.join(input_dir, prefix + '.qids.npy'))
  return (q, a, q_overlap, a_overlap, y, qids)

def main(argv):
  if (len(argv) != 4):
    print('usage: run_nnet.py inputdir outputdir train/test/all')
    exit(1)
  input_dir = argv[1]
  output_dir = argv[2]
  do_train = (argv[3] == 'train' or argv[3] == 'all')
  do_test = (argv[3] == 'test' or argv[3] == 'all')
 
  # init random seed
  numpy.random.seed(100)
  numpy_rng = numpy.random.RandomState(123)

  # prepare output dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  x_q = T.matrix('q')
  x_a = T.matrix('a')
  x_q_overlap = T.matrix('q_overlap')
  x_a_overlap = T.matrix('a_overlap')
  y = T.vector('y')
  batch_x_q = T.matrix('batch_x_q')
  batch_x_a = T.matrix('batch_x_a')
  batch_x_q_overlap = T.matrix('batch_x_q_overlap')
  batch_x_a_overlap = T.matrix('batch_x_a_overlap')
  batch_y = T.vector('batch_y')

  if (do_train):
    # set hyper parameters
    ZEROUT_DUMMY_WORD = True
    n_outs = 2
    learning_rate = 0.1
    max_norm = 0
    print 'batch_size', batch_size
    print 'n_epochs', n_epochs
    print 'learning_rate', learning_rate
    print 'max_norm', max_norm
    # load data
    print "Running training with the data in {}".format(input_dir)
    q_train, a_train, q_overlap_train, a_overlap_train, y_train, qids_train = load_data(input_dir, 'train')
    q_max_sent_size = q_train.shape[1]
    a_max_sent_size = a_train.shape[1]
    q_dev, a_dev, q_overlap_dev, a_overlap_dev, y_dev, qids_dev = load_data(input_dir, 'dev')
    dev_size = q_dev.shape[0]
    sample_idx = numpy.arange(dev_size)
    numpy.random.shuffle(sample_idx)
    sample_idx = sample_idx[:min(n_dev_batch * batch_size, dev_size)]
    q_dev = q_dev[sample_idx]
    a_dev = a_dev[sample_idx]
    y_dev = y_dev[sample_idx]
    qids_dev = qids_dev[sample_idx]
    print 'y_train', numpy.unique(y_train, return_counts=True)
    print 'y_dev', numpy.unique(y_dev, return_counts=True)
    print 'q_train', q_train.shape
    print 'q_dev', q_dev.shape
    print 'a_train', a_train.shape
    print 'a_dev', a_dev.shape
    print 'max_sent_size', numpy.max(a_train)
    print 'min_sent_size', numpy.min(a_train) 
    q_overlap_dev = q_overlap_dev[sample_idx]
    a_overlap_dev = a_overlap_dev[sample_idx]  
    dummy_word_id = numpy.max(a_overlap_train) 
  if (do_test):
    q_test, a_test, q_overlap_test, a_overlap_test, y_test, qids_test = load_data(input_dir, 'test')
    q_max_sent_size = q_test.shape[1]
    a_max_sent_size = a_test.shape[1]
    dummy_word_id = numpy.max(a_overlap_test)
    
  # number of dimmension for overlapping feature (the 0,1,2 features)
  ndim = 5
  print "Generating random vocabulary for word overlap indicator features with dim", ndim
  vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, ndim) * 0.25
  vocab_emb_overlap[-1] = 0
  vocab_emb_overlap = vocab_emb_overlap.astype(numpy.float32)

  # Load word2vec embeddings
  fname = os.path.join(input_dir, 'emb_aquaint+wiki.txt.gz.ndim=50.bin.npy')
  print "Loading word embeddings from", fname
  vocab_emb = numpy.load(fname)
  ndim = vocab_emb.shape[1]
  print "Word embedding matrix size:", vocab_emb.shape
  vocab_emb = vocab_emb.astype(numpy.float32)

  # build network
  nnet = deep_qa_net(batch_size, vocab_emb, vocab_emb_overlap, 
                            q_max_sent_size, a_max_sent_size, numpy_rng,
                            x_q, x_q_overlap,
                            x_a, x_a_overlap,
                            100, 1, 1, 1, 0.5)
  if do_train:
    nnet_fname = os.path.join(output_dir, 'nnet.dat')
    print "Saving to", nnet_fname
    cPickle.dump([nnet], open(nnet_fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

  print nnet
  params = nnet.params
  total_params = sum([numpy.prod(param.shape.eval()) for param in params])
  print 'Total params number:', total_params

  predictions = nnet.layers[-1].y_pred
  predictions_prob = nnet.layers[-1].p_y_given_x[:,-1]

  inputs_pred = [batch_x_q,
                 batch_x_a,
                 batch_x_q_overlap,
                 batch_x_a_overlap,
                 # batch_x,
                 ]

  givens_pred = {x_q: batch_x_q,
                 x_a: batch_x_a,
                 x_q_overlap: batch_x_q_overlap,
                 x_a_overlap: batch_x_a_overlap,
                 # x: batch_x
                 }

  inputs_train = [batch_x_q,
                 batch_x_a,
                 batch_x_q_overlap,
                 batch_x_a_overlap,
                 # batch_x,
                 batch_y,
                 ]

  givens_train = {x_q: batch_x_q,
                 x_a: batch_x_a,
                 x_q_overlap: batch_x_q_overlap,
                 x_a_overlap: batch_x_a_overlap,
                 # x: batch_x,
                 y: batch_y}

  if (do_train):
    cost = nnet.layers[-1].training_cost(T.cast(y, 'int32'))
    if (regularize):
      #### L2 regularization
      L2_word_emb = 1e-4
      L2_conv1d = 3e-5
      # L2_softmax = 1e-3
      L2_softmax = 1e-4
      print "Regularizing nnet weights"
      for w in nnet.weights:
        L2_reg = 0.
        if w.name.startswith('W_emb'):
          L2_reg = L2_word_emb
        elif w.name.startswith('W_conv1d'):
          L2_reg = L2_conv1d
        elif w.name.startswith('W_softmax'):
          L2_reg = L2_softmax
        elif w.name == 'W':
          L2_reg = L2_softmax
        print w.name, L2_reg
        cost += T.sum(w**2) * L2_reg

    updates = sgd_trainer.get_adadelta_updates(cost, params, rho=0.95, eps=1e-6, max_norm=max_norm, word_vec_name='W_emb')
    train_fn = theano.function(inputs=inputs_train,
                               outputs=cost,
                               updates=updates,
                               givens=givens_train)

  pred_fn = theano.function(inputs=inputs_pred,
                            outputs=predictions,
                            givens=givens_pred)

  pred_prob_fn = theano.function(inputs=inputs_pred,
                            outputs=predictions_prob,
                            givens=givens_pred)

  def predict_batch(batch_iterator):
    preds = numpy.hstack([pred_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
    return preds[:batch_iterator.n_samples]

  def predict_prob_batch(batch_iterator):
    preds = numpy.hstack([pred_prob_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
    return preds[:batch_iterator.n_samples]

  def map_score(qids, labels, preds):
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
      qid2cand[qid].append((pred, label))

    average_precs = []
    for qid, candidates in qid2cand.iteritems():
      average_prec = 0
      running_correct_count = 0
      for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
        if label > 0:
          running_correct_count += 1
          average_prec += float(running_correct_count) / i
      average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score

  if (do_train):
    train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_train, a_train, q_overlap_train, a_overlap_train, y_train], batch_size=batch_size, randomize=True)
    dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_dev, a_dev, q_overlap_dev, a_overlap_dev, y_dev], batch_size=batch_size, randomize=False)
    print "Zero out dummy word:", ZEROUT_DUMMY_WORD
    if ZEROUT_DUMMY_WORD:
      W_emb_list = [w for w in params if w.name == 'W_emb']
      zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])


    # weights_dev = numpy.zeros(len(y_dev))
    # weights_dev[y_dev == 0] = weights_data[0]
    # weights_dev[y_dev == 1] = weights_data[1]
    # print weights_dev

    best_dev_acc = -numpy.inf
    epoch = 0
    timer_train = time.time()
    no_best_dev_update = 0
    num_train_batches = len(train_set_iterator)
    while epoch < n_epochs:
        timer = time.time()
        for i, (x_q, x_a, x_q_overlap, x_a_overlap, y) in enumerate(tqdm(train_set_iterator), 1):
            train_fn(x_q, x_a, x_q_overlap, x_a_overlap, y)

            # Make sure the null word in the word embeddings always remains zero
            if ZEROUT_DUMMY_WORD:
              zerout_dummy_word()

            if i % n_iter_per_val == 0 or i == num_train_batches:
              y_pred_dev = predict_prob_batch(dev_set_iterator)
              # # dev_acc = map_score(qids_dev, y_dev, predict_prob_batch(dev_set_iterator)) * 100
              dev_acc = metrics.roc_auc_score(y_dev, y_pred_dev) * 100
              if dev_acc > best_dev_acc:
                #y_pred = predict_prob_batch(test_set_iterator)
                #test_acc = map_score(qids_test, y_test, y_pred) * 100

                #print('epoch: {} batch: {} dev auc: {:.4f}; test map: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc, test_acc, best_dev_acc))
                print('epoch: {} batch: {} dev auc: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc, best_dev_acc))
                best_dev_acc = dev_acc
                best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                no_best_dev_update = 0

        if no_best_dev_update >= early_stop_epochs:
          print "Quitting after of no update of the best score on dev set", no_best_dev_update
          break

        print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))
        param_fname = os.path.join(output_dir, 'parameters_epoch={}_acc={}.dat'.format(epoch, best_dev_acc))
        cPickle.dump(best_params, open(param_fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        epoch += 1
        no_best_dev_update += 1
    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))

  param_fname = os.path.join(output_dir, 'best_params.dat')
  if (do_train):
    for i, param in enumerate(best_params):
      params[i].set_value(param, borrow=True)
    cPickle.dump(best_params, open(param_fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    print('dumpping params to {}'.format(param_fname))
  if (do_test):
    best_params = cPickle.load(open(param_fname, 'rb'))
    for i, param in enumerate(best_params):
      params[i].set_value(param, borrow=True)    
    test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, y_test], batch_size=batch_size, randomize=False)
    # do test
    timer_test = time.time()
    print "Number of QA pairs: ", len(q_test)
    y_pred_test = predict_prob_batch(test_set_iterator)
    print('Testing took: {:.4f} seconds'.format(time.time() - timer_test))
    auc = metrics.roc_auc_score(y_test, y_pred_test)
    print("AUC on test data: {}".format(auc))
    test_acc = map_score(qids_test, y_test, y_pred_test) * 100 
    print "Running trec_eval script..."
    N = len(y_pred_test)
    df_submission = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
    df_submission['qid'] = qids_test
    df_submission['iter'] = 0
    df_submission['docno'] = numpy.arange(N)
    df_submission['rank'] = 0
    df_submission['sim'] = y_pred_test
    df_submission['run_id'] = 'nnet'
    df_submission.to_csv(os.path.join(output_dir, 'submission.txt'), header=False, index=False, sep=' ')
    df_gold = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
    df_gold['qid'] = qids_test
    df_gold['iter'] = 0
    df_gold['docno'] = numpy.arange(N)
    df_gold['rel'] = y_test
    df_gold.to_csv(os.path.join(output_dir, 'gold.txt'), header=False, index=False, sep=' ')
    subprocess.call("/bin/sh run_eval.sh '{}'".format(output_dir), shell=True)

if __name__ == '__main__':
  main(sys.argv)
