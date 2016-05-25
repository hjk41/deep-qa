from datetime import datetime
from sklearn import metrics
from theano import tensor as T, printing
import cPickle
import numpy
import os
import sys
import theano
import time
from collections import defaultdict
import subprocess
import pandas as pd
import argparse
from tqdm import tqdm

import nn_layers
import sgd_trainer

import warnings
warnings.filterwarnings("ignore")  # TODO remove

### THEANO DEBUG FLAGS
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = 'high'

n_epochs = 100
batch_size = 50
n_dev_batch = 200
n_iter_per_val = 1000
n_conv_kernels = 400
early_stop_epochs = 10
regularize = False
dropout_rate = 0.5
sgd_learning_rate=1
embed_on_cpu = True
use_overlap = True
if not use_overlap:
  theano.config.on_unused_input = 'warn'
# init random seed
numpy.random.seed(100)
numpy_rng = numpy.random.RandomState(123)

def conv_layer(batch_size,
    embedding,                # embedding dictionary, from id to vector, static
    embedding_overlap,        # embedding matrix for the overlap feature, will be updated
    numpy_randg,              # numpy random number generator
    x,                        # intput matrix, each sentense is a row, each element is index
    x_overlap,                # overlap matrix
    filter_width = 4,         # filter widths to use, currently 3
    n_conv_kern = 100,        # number of convolution kernels, currently 100
    n_input_channel = 1):
  '''
  Build the convolution layer that convs the q and a
  
  Returns a network with input (x, x_overlap)
  '''
  if (embed_on_cpu):
    lookup_table_words = nn_layers.PadLayer(pad=filter_width-1, axis=2)
  else:
    # perform lookup and then pad the matrix so we can later use it in conv
    lookup_table_words = nn_layers.LookupTableFastStatic(W=embedding, pad=filter_width-1)
  if (use_overlap):
    lookup_table_overlap = nn_layers.LookupTableFast(W=embedding_overlap, pad=filter_width-1)
    # concatenate the outputs of words and overlap into one
    # now the output dim is embedding+embedding_overlap, currently 50+5
    lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap])
    ndim = embedding.shape[1] + embedding_overlap.shape[1]
  else:
    lookup_table = lookup_table_words
    ndim = embedding.shape[1]
  #input_shape = (batch_size, n_input_channel, max_sent_length + 2*(max(filter_widths)-1), ndim)
  filter_shape = (n_conv_kern, n_input_channel, filter_width, ndim)
  conv = nn_layers.Conv2dLayer(rng=numpy_randg, filter_shape=filter_shape)
  non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0], activation=T.tanh)
  nnet = nn_layers.FeedForwardNet(layers=[lookup_table, conv, non_linearity])
  if (embed_on_cpu):
    inputx = x
  else:
    inputx = T.cast(x, 'int32')
  if (use_overlap):
    nnet.set_input([inputx, T.cast(x_overlap, 'int32')])
  else:
    nnet.set_input([inputx])
  return nnet

def deep_qa_net(batch_size,
    embedding,                # embedding dictionary, from id to vector, static
    embedding_overlap,        # embedding matrix for the overlap feature, will be updated
    numpy_randg,              # numpy random number generator
    x_q,                      # input symbols
    x_q_overlap,
    x_a,
    x_a_overlap,
    n_conv_kern = 100,        # number of convolution kernels, currently 100
    n_input_channel = 1,
    q_k_max = 1,              # output of max polling, currently 1
    a_k_max = 1,              
    dropout = 0.5,
    dropout_on = None):    
  '''
  Builds deepQA network

  batch_size:               int, size of the batch
  max_sent_length:          int, maximum length of sentense
  embedding:                numpy array, embedding dictionary, from id to vector, static
  embedding_overlap:        numpy array, embedding matrix for the overlap feature, will be updated
  n_conv_kern               int, number of convolution kernels, currently 100
  n_input_channel           int, number of input channel for convolution, currently 1
  '''
  ## question conv
  nnet_q = conv_layer(batch_size, 
                      embedding, embedding_overlap, numpy_randg, 
                      x_q, x_q_overlap,
                      3, n_conv_kern, 1)
  ## answer conv
  nnet_a = conv_layer(batch_size, 
                      embedding, embedding_overlap, numpy_randg, 
                      x_a, x_a_overlap,
                      3, n_conv_kern, 1)
  output_q = nnet_q.output
  output_a = nnet_a.output
  layers = [nnet_q, nnet_a]
  if (dropout != 0.0):
    dropout_q = nn_layers.DropoutLayer(rng=numpy_randg, p=dropout_rate, dropout_on=dropout_on) 
    dropout_a = nn_layers.DropoutLayer(rng=numpy_randg, p=dropout_rate, dropout_on=dropout_on) 
    dropout_q.set_input(nnet_q.output) 
    dropout_a.set_input(nnet_a.output) 
    layers.append(dropout_q)
    layers.append(dropout_a)
    output_q = dropout_q.output
    output_a = dropout_a.output
  attentive_similarity = nn_layers.PairwiseAttentivePollingSimilarity(n_conv_kern)
  attentive_similarity.set_input((output_q, output_a))
  layers.append(attentive_similarity)
  #classifier = nn_layers.CosineSimilarityLoss(m = 0.3)
  classifier = nn_layers.SingleLR(m = 0.3)
  classifier.set_input(attentive_similarity.output.reshape([attentive_similarity.output.shape[0]]))
  layers.append(classifier)
  nnet = nn_layers.FeedForwardNet(layers=layers,
                                          name="nnet")
  return nnet

def to1hot(data):
  b = numpy.zeros((data.shape[0], data.shape[1], n_overlap_choices))
  for i in range(data.shape[0]):
      for j in range(data.shape[1]):
        b[i, j, data[i, j]] = 1.0
  return b

def embed(data, emb):
  return emb[data.astype(numpy.int).flatten()].reshape((data.shape[0], 1, data.shape[1], emb.shape[1]))

def load_data(input_dir):
  q = numpy.load(os.path.join(input_dir, 'questions.npy')).astype(numpy.float32)
  a = numpy.load(os.path.join(input_dir, 'answers.npy')).astype(numpy.float32)
  q_overlap = numpy.load(os.path.join(input_dir, 'q_overlap_indices.npy')).astype(numpy.float32)
  a_overlap = numpy.load(os.path.join(input_dir, 'a_overlap_indices.npy')).astype(numpy.float32)
  y = numpy.load(os.path.join(input_dir, 'labels.npy')).astype(numpy.float32)
  qids = numpy.load(os.path.join(input_dir, 'qids.npy'))
  lineids = numpy.load(os.path.join(input_dir, 'lineids.npy'))
  return (q, a, q_overlap, a_overlap, y, qids, lineids)

def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('mode', type=str, help='train/test/all')
  parser.add_argument('-t', '--train', help='training set dir')
  parser.add_argument('-d', '--validation', help='validation set dir')
  parser.add_argument('-e', '--test', help='test set dir')
  parser.add_argument('-o', '--output', help='output result file')
  parser.add_argument('-b', '--embed', help='embedding dir')
  parser.add_argument('-m', '--model', help='model parameter file')
  parser.add_argument('-c', '--trec', type=bool, default=True, help='whether to run trec eval script')
  parser.add_argument('-g', '--debug', type=bool, default=False, help='enable ptvsd debugging')
  args = parser.parse_args()

  if (args.debug):
    import ptvsd
    ptvsd.enable_attach(secret='secret')
    ptvsd.wait_for_attach()
  
  do_train = (args.mode == 'train' or args.mode == 'all')
  do_test = (args.mode == 'test' or args.mode == 'all')
  if (not do_train) and (not do_test):
    print('mode must be train, test or all')
    exit(1)
  if (do_train and (not args.train)):
    print('training set must be specified')
    exit(1)
  if (do_train and (not args.validation)):
    print('validation set must be specified')
    exit(1)
  if (do_test and (not args.test)):
    print('test set must be specified')
    exit(1)
  if (do_test and (not args.output)):
    print('result file must be specified')
    exit(1)
  if (do_train or do_test) and (not args.embed):
    print('embedding file must be specified')
    exit(1)
  if (do_train or do_test) and (not args.model):
    print('model file must be specified')
    exit(1)

  # prepare output dir
  if (args.model):
    dir, _ = os.path.split(args.model)
    if (not os.path.exists(dir)):
      os.makedirs(dir)
  if (args.output):
    dir, _ = os.path.split(args.output)
    if (not os.path.exists(dir)):
      os.makedirs(dir)

  if (embed_on_cpu):
    Q = T.tensor4('q')
    A = T.tensor4('a')
  else:
    Q = T.matrix('q')
    A = T.matrix('a')
  Q_OVERLAP = T.matrix('q_overlap')
  A_OVERLAP = T.matrix('a_overlap')
  Y = T.vector('y')
  DROPOUT_ON = T.scalar('dropout_on')

  # number of dimmension for overlapping feature (the 0,1,2 features)
  ndim = 5
  print "Generating random vocabulary for word overlap indicator features with dim", ndim
  vocab_emb_overlap = numpy_rng.randn(3, ndim) * 0.25
  vocab_emb_overlap[-1] = 0
  vocab_emb_overlap = vocab_emb_overlap.astype(numpy.float32)

  # Load word2vec embeddings
  fname = os.path.join(args.embed, 'emb.npy')
  print "Loading word embeddings from", fname
  vocab_emb = numpy.load(fname)
  print "Word embedding matrix size:", vocab_emb.shape
  # append a vector for unknown words, and a vector for empty words
  embed_dim = vocab_emb.shape[1]
  unknown_emb = numpy.ones(embed_dim)
  empty_emb = numpy.zeros(embed_dim)
  vocab_emb = numpy.concatenate([vocab_emb, numpy.array([unknown_emb, empty_emb])], axis=0)
  vocab_emb = vocab_emb.astype(numpy.float32)

  if (do_train):
    # set hyper parameters
    ZEROUT_DUMMY_WORD = True
    learning_rate = 0.1
    max_norm = 0
    print 'batch_size', batch_size
    print 'n_epochs', n_epochs
    print 'learning_rate', learning_rate
    print 'max_norm', max_norm
    # load data
    print "Running training with train={}, validation={}".format(args.train, args.validation)
    q_train, a_train, q_overlap_train, a_overlap_train, y_train, qids_train, _ = load_data(args.train)
    # shuffle training data
    sample_idx = numpy.arange(q_train.shape[0])
    numpy.random.shuffle(sample_idx)
    q_train = q_train[sample_idx]
    a_train = a_train[sample_idx]
    q_overlap_train = q_overlap_train[sample_idx]
    a_overlap_train = a_overlap_train[sample_idx]
    y_train = y_train[sample_idx]
    qids_train = qids_train[sample_idx]
    # subsample the dev set
    q_dev, a_dev, q_overlap_dev, a_overlap_dev, y_dev, qids_dev, _ = load_data(args.validation)
    dev_size = q_dev.shape[0]
    sample_idx = numpy.arange(dev_size)
    numpy.random.shuffle(sample_idx)
    sample_idx = sample_idx[:min(n_dev_batch * batch_size, dev_size)]
    q_dev = q_dev[sample_idx]
    a_dev = a_dev[sample_idx]
    y_dev = y_dev[sample_idx]
    qids_dev = qids_dev[sample_idx]
    q_overlap_dev = q_overlap_dev[sample_idx]
    a_overlap_dev = a_overlap_dev[sample_idx]  
    print 'y_train', numpy.unique(y_train, return_counts=True)
    print 'y_dev', numpy.unique(y_dev, return_counts=True)
    print 'q_train', q_train.shape
    print 'q_dev', q_dev.shape
    print 'a_train', a_train.shape
    print 'a_dev', a_dev.shape
  if (do_test):
    q_test, a_test, q_overlap_test, a_overlap_test, y_test, qids_test, lineids_test = load_data(args.test)
    q_max_sent_size = q_test.shape[1]
    a_max_sent_size = a_test.shape[1]

  # build network
  nnet = deep_qa_net(batch_size, vocab_emb, vocab_emb_overlap, 
                            numpy_rng,
                            Q, Q_OVERLAP,
                            A, A_OVERLAP,
                            n_conv_kernels, 1, 1, 1, dropout_rate,
                            DROPOUT_ON)
  #if do_train:
  #  nnet_fname = os.path.join(output_dir, 'nnet.dat')
  #  print "Saving to", nnet_fname
  #  cPickle.dump([nnet], open(nnet_fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

  print nnet
  params = nnet.params
  total_params = sum([numpy.prod(param.shape.eval()) for param in params])
  print 'Total params number:', total_params

  predictions = nnet.layers[-1].y_pred
  predictions_prob = nnet.layers[-1].p_y_given_x

  #inputs_pred = [batch_x_q,
  #               batch_x_a,
  #               batch_x_q_overlap,
  #               batch_x_a_overlap,
  #               # batch_x,
  #               ]

  #givens_pred = {x_q: batch_x_q,
  #               x_a: batch_x_a,
  #               x_q_overlap: batch_x_q_overlap,
  #               x_a_overlap: batch_x_a_overlap,
  #               # x: batch_x
  #               }

  #inputs_train = [batch_x_q,
  #               batch_x_a,
  #               batch_x_q_overlap,
  #               batch_x_a_overlap,
  #               # batch_x,
  #               batch_y,
  #               ]

  #givens_train = {x_q: batch_x_q,
  #               x_a: batch_x_a,
  #               x_q_overlap: batch_x_q_overlap,
  #               x_a_overlap: batch_x_a_overlap,
  #               # x: batch_x,
  #               y: batch_y}

  if (do_train):
    cost = nnet.layers[-1].training_cost(Y)
    if (regularize):
      #### L2 regularization
      L2_word_emb = 1e-2
      L2_conv1d = 1e-2
      L2_softmax = 3e-3
      L2_pairwise = 1e-3
      print "Regularizing nnet weights"
      for w in nnet.weights:
        L2_reg = 0.001
        if w.name.startswith('W_emb'):
          L2_reg = L2_word_emb
        elif w.name.startswith('W_conv1d'):
          L2_reg = L2_conv1d
        elif w.name.startswith('W_softmax'):
          L2_reg = L2_softmax
        elif w.name.startswith('W_pairwise'):
          L2_reg = L2_pairwise
        elif w.name == 'W':
          L2_reg = L2_softmax
        print w.name, L2_reg
        cost += T.sum(w**2) * L2_reg

    updates = sgd_trainer.get_adadelta_updates(cost, params, rho=0.95, eps=1e-6, max_norm=max_norm, word_vec_name='W_emb')
    #updates = sgd_trainer.get_sgd_updates(cost, params, learning_rate=sgd_learning_rate, max_norm=None, rho=0.95)
    train_fn = theano.function(inputs=[Q, A, Q_OVERLAP, A_OVERLAP, Y, DROPOUT_ON],
                               outputs=cost,
                               updates=updates)

  pred_fn = theano.function(inputs=[Q, A, Q_OVERLAP, A_OVERLAP, DROPOUT_ON],
                            outputs=predictions)

  pred_prob_fn = theano.function(inputs=[Q, A, Q_OVERLAP, A_OVERLAP, DROPOUT_ON],
                            outputs=predictions_prob)

  loss_fn = theano.function(inputs=[Q, A, Q_OVERLAP, A_OVERLAP, Y, DROPOUT_ON],
                            outputs=cost)

  def predict_batch(batch_iterator):
    if (embed_on_cpu):
      preds = numpy.hstack([pred_fn(embed(batch_x_q, vocab_emb), embed(batch_x_a, vocab_emb), 
                                    batch_x_q_overlap, batch_x_a_overlap, 0.0) 
                                    for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
    else:
      preds = numpy.hstack([pred_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, 0.0) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
    return preds[:batch_iterator.n_samples]

  def predict_prob_batch(batch_iterator):
    if (embed_on_cpu):
      preds = numpy.hstack([pred_prob_fn(embed(batch_x_q, vocab_emb), embed(batch_x_a, vocab_emb), 
                                    batch_x_q_overlap, batch_x_a_overlap, 0.0) 
                                    for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
    else:
      preds = numpy.hstack([pred_prob_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, 0.0) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
    return preds[:batch_iterator.n_samples]

  def loss_batch(batch_iterator):
    if (embed_on_cpu):
      loss = numpy.sum(numpy.hstack([loss_fn(embed(batch_x_q, vocab_emb), embed(batch_x_a, vocab_emb), 
                                    batch_x_q_overlap, batch_x_a_overlap, y, 1.0) 
                                    for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, y in batch_iterator]))
    else:
      loss = numpy.sum(numpy.hstack([loss_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, y, 1.0) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, y in batch_iterator]))
    return loss

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
    train_set_iterator_norandom = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_train, a_train, q_overlap_train, a_overlap_train, y_train], batch_size=batch_size, randomize=False)
    #print "Zero out dummy word:", ZEROUT_DUMMY_WORD
    #if ZEROUT_DUMMY_WORD:
    #  W_emb_list = [w for w in params if w.name == 'W_emb']
    #  zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])


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
        train_loss = 0.0
        for i, (x_q, x_a, x_q_overlap, x_a_overlap, y) in enumerate(tqdm(train_set_iterator), 1):
            if (embed_on_cpu):
              train_loss = train_fn(embed(x_q, vocab_emb), embed(x_a, vocab_emb), 
                                    x_q_overlap, x_a_overlap, y, 1.0)
            else:
              train_loss = train_fn(x_q, x_a, x_q_overlap, x_a_overlap, y, 1.0)
            #best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]

            # Make sure the null word in the word embeddings always remains zero
            #if ZEROUT_DUMMY_WORD:
            #  zerout_dummy_word()

            if i % n_iter_per_val == 0 or i == num_train_batches:
            #if True:
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
        # evaluate on training data
        #y_pred_train = predict_prob_batch(train_set_iterator_norandom)
        #train_acc = metrics.roc_auc_score(y_train, y_pred_train) * 100
        # train_loss = loss_batch(train_set_iterator_norandom)
        #print('epoch: {} train auc: {:.4f} train loss:{}'.format(epoch, train_acc, train_loss))

        if no_best_dev_update >= early_stop_epochs:
          print "Quitting after of no update of the best score on dev set", no_best_dev_update
          break

        print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))
        cPickle.dump(best_params, open(args.model, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        epoch += 1
        no_best_dev_update += 1
    print('Training took: {:.4f} seconds'.format(time.time() - timer_train))

  if (do_train):
    for i, param in enumerate(best_params):
      params[i].set_value(param, borrow=True)
    print('dumpping params to {}'.format(args.model))
    cPickle.dump(best_params, open(args.model, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
  if (do_test):
    print('doing testing...')
    best_params = cPickle.load(open(args.model, 'rb'))
    for i, param in enumerate(best_params):
      params[i].set_value(param, borrow=True)    
    test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, y_test], batch_size=batch_size, randomize=False)
    # do test
    timer_test = time.time()
    print "Number of QA pairs: ", len(q_test)
    y_pred_test = predict_prob_batch(test_set_iterator)
    print('Testing took: {:.4f} seconds'.format(time.time() - timer_test))
    try:
      auc = metrics.roc_auc_score(y_test, y_pred_test)
      print("AUC on test data: {}".format(auc))
    except:
      pass
    #test_acc = map_score(qids_test, y_test, y_pred_test) * 100 
    N = len(y_pred_test)
    df_submission = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
    df_submission['qid'] = qids_test
    df_submission['iter'] = 0
    df_submission['docno'] = numpy.arange(N)
    df_submission['rank'] = 0
    df_submission['sim'] = y_pred_test
    if (len(lineids_test) != len(qids_test)):
      print('test lineids size={}, using dummy lineids'.format(len(lineids_test)))
      df_submission['run_id'] = 'nnet'
    else:
      df_submission['run_id'] = lineids_test
    df_submission.to_csv(args.output, header=False, index=False, sep=' ')
    df_gold = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
    df_gold['qid'] = qids_test
    df_gold['iter'] = 0
    df_gold['docno'] = numpy.arange(N)
    df_gold['rel'] = y_test
    df_gold.to_csv(args.output + '.gold', header=False, index=False, sep=' ')
    if (args.trec):
      print "Running trec_eval script..."
      subprocess.call("/bin/sh run_eval.sh '{}' '{}'".format(args.output, args.output + '.gold'), shell=True)

if __name__ == '__main__':
  main(sys.argv)
