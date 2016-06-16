#!/usr/bin/python

'''
Utility to load and convert datasets
'''
import argparse
import logging
import numpy
import os
import pickle
import re
import sys

from utils import setup_logger, enable_ptvsd

MAX_SENT_LENGTH = 70


class QADatasetRaw():
    '''
    Represents a dataset in answer sentense selection problem.
    
    This is raw data, meaning q and a are stored as words, not index.

    Fields:
        qids:    int(num_samples,)    question ids for each sample
        q:    string(num_samples, idx<MAX_SENT_LENGTH)    word array
        a:    string(num_samples, idx<MAX_SENT_LENGTH)    word array
        labels:    float(num_samples,)    label for each sample
    '''
    def __init__(self, qids, q, a, labels):
        self.qids = qids
        self.q = q
        self.a = a
        self.labels = labels


class QADataset():
    '''
    Represents a dataset in answer sentense selection problem.
    
    This is parsed data, q and a are indexes.

    Fields:
        qids:    int(num_samples,)    question ids for each sample
        q:    numpy.array(num_samples, MAX_SENT_LENGTH)    index array
        a:    numpy.array(num_samples, MAX_SENT_LENGTH)    index array
        labels:    numpy.array(num_samples,)    label for each sample
    '''
    def __init__(self, qids, q, a, labels, q_overlap, a_overlap):
        self.qids = qids
        self.q = q
        self.a = a
        self.labels = labels
        self.q_overlap = q_overlap
        self.a_overlap = a_overlap


def __load_trec(fname, max_sent_length=MAX_SENT_LENGTH, skip_long_sent=False):
    '''
    Load file in the form of TREC xml, that is:
    <QAPairs id='1'>
        <question>
            What    is  a   mouse   ?
        </question>
        <positive>
            Mouse   is  a   mammal  ,   ...
        </positive>
        <negative>
            Dog is  a   kind    of  ...
        </negative>
    </QAPairs>

    Params:
        fname:    string    path to the xml file
        max_sent_length:    int    maximum sentense length
        skip_long_sent:    bool    whether to skip long sentenses

    Returns:
        QADataset
    '''
    logging.info('Loading file {}, '
                 'max sentense length {}, '
                 'skip long sentense {}'
                 .format(fname, max_sent_length, skip_long_sent))
    qids, questions, answers, labels = [], [], [], []
    num_long_sentense = 0
    prev = ''
    question2qid = {}
    curr_qid = 0
    for i, line in enumerate(open(fname)):
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
            if len(answer) > max_sent_length:
                num_long_sentense += 1
                if (skip_long_sent == True):
                    continue
                else:
                    answer = answer[:max_sent_length]
            labels.append(label)
            answers.append(answer)
            questions.append(question)
            qids.append(qid)
    logging.info('Loading complete')
    logging.info('Number of long senteses: {}'.format(num_long_sentense))
    return QADatasetRaw(qids, questions, answers, labels)


def __passage2list(psg):
    '''
    Split the passage into a list of words.
    Punctuations will be split from words if there is no whitespace between a 
    puctuation and a word.

    Params:
        psg:    string    string storing a passage

    Returns:
        list of words in the passaage
    '''
    list = []
    pos = 0
    wordStart = pos
    isFirstChar = True
    lastIsPunc = False
    n = len(psg)
    alphabet = 'abcdefghijklmnopqrstuvwxyz1234567890-_'
    while (pos < n):
        c = psg[pos]
        if c == ' ':
            if (not isFirstChar):
                list.append(psg[wordStart:pos])
                lastIsPunc = False
                isFirstChar = True
        elif (c not in alphabet):
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


def __load_tsv(fname, max_sent_length=MAX_SENT_LENGTH,
             skip_long_sent=False, treat_good='remove',
             has_header=False):
    '''
    Load a dataset in the form of Tab Seperated Vector (TSV) file

    The TSV file is stored in the format of:
        Query   Url     PassageID       Passage Rating1 Rating2
    In which Rating2 is used, and its values is in {'Bad', 'Good', 'Perfect'}

    Params:
        fname:    string    path of the tsv file
        max_sent_length:    int     max length of a sentense
        skip_long_sent:    bool    whether long sentense should be skipped or trimmed
        treat_good:    string    how to treat 'Good' ratings, can be
                                 ['remove', 'positive', 'negative']
        has_header:    bool    does the file has header?

    Returns:
        QADataset
    '''
    assert treat_good in ['remove', 'positive', 'negative']
    logging.info('Loading file {}, '
                 'max sentense length {}, '
                 'skip long sentense {}, '
                 'treat good as {}, '
                 'has header: {}'
                 .format(fname, max_sent_length, skip_long_sent, 
                         treat_good, has_header))
    qids, questions, answers, labels = [], [], [], []
    lineids = []
    curr_qid = 0
    num_long_sents = 0
    question2qid = {}
    file = open(fname)
    if (has_header):
        header = file.readline()
        logging.info('header in TSV file: {}'.format(header))
    for i, line in enumerate(file):
        line = line.strip().lower()
        # Query     Url         PassageID             Passage Rating1 Rating2
        qupprr=line.split('\t')
        if len(qupprr) != 6:
            logging.exception('error parsing line {}\nline:{}\n'.format(i+1, line))
        q = qupprr[0].lower()
        if not q in question2qid:
            question2qid[q] = curr_qid
            qid = curr_qid
            curr_qid += 1
        else:
            qid = question2qid[q]
        question = __passage2list(q)            ### should we convert to lower case?
        answer = __passage2list(qupprr[3].lower())
        if len(answer) > max_sent_length:
            num_long_sents += 1
            if(skip_long_sent):
                continue
            else:
                answer = answer[:max_sent_length]
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
    logging.info('Loading complete')
    logging.info('Number of long senteses: {}'.format(num_long_sents))
    return QADatasetRaw(qids, questions, answers, labels)


def __convert_dataset(inputfile, outputdir, format='trec'):
    '''
    Convert a dataset into the feature files we need, and store
    the result in outdir.

    The output files includes:
        qids.pickle
        questions.pickle
        answers.pickle
        labels.pickle

    Params:
        intputfile:    string    path of input file
        outputdir:    string    path of the output directory
        format:    string    ['trec','tsv']
    '''
    assert format in ['trec', 'tsv']
    logging.info('Converting file {}'.format(inputfile))
    if (format == 'trec'):
        dataset = __load_trec(inputfile)
    else:
        dataset = __load_tsv(inputfile, treat_good='remove')
    if (not os.path.exists(outputdir)):
        os.makedirs(outputdir)
    logging.info('Dumpping files to {}'.format(outputdir))
    pickle.dump(dataset.qids, open(os.path.join(outputdir, 'qids.pickle'), 'wb'))
    pickle.dump(dataset.q, open(os.path.join(outputdir, 'questions.pickle'), 'wb'))
    pickle.dump(dataset.a, open(os.path.join(outputdir, 'answers.pickle'), 'wb'))
    pickle.dump(dataset.labels, open(os.path.join(outputdir, 'labels.pickle'), 'wb'))
    logging.info('Done!')


def __load_raw_dataset(datadir):
    '''
    Load an answer selection dataset

    Params:
        datadir:    string    path to the directory storing dataset files

    Returns:
        QADatasetRaw
    '''
    return QADatasetRaw(pickle.load(open(os.path.join(datadir, 'qids.pickle'), 'rb')),
                     pickle.load(open(os.path.join(datadir, 'questions.pickle'), 'rb')),
                     pickle.load(open(os.path.join(datadir, 'answers.pickle'), 'rb')),
                     pickle.load(open(os.path.join(datadir, 'labels.pickle'), 'rb')))


def __compute_overlap_features_given_word_vector(questions, answers, 
                                                 word2df=None, stoplist=None):
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
        for w in word_overlap:                ### should we count the word frequency?
            df_overlap += word2df[w]
        #total_dfs = 0.0
        #for w in q_set:
        #    total_dfs += word2df[w]
        #for w in a_set:
        #    total_dfs += word2df[w]
        # df_overlap = total_overlap_IDF / (words_in_q + words_in_a)
        df_overlap /= (len(q_set) + len(a_set))
        #df_overlap /= total_dfs

        feats_overlap.append(numpy.array([overlap, df_overlap]))
    return numpy.array(feats_overlap)


def __compute_overlap_idx_given_word_vector(questions, answers, stoplist, 
                        max_q_length, max_a_length):
    '''
    Compute overlap feature of q and a.

    For each pair of q and a, output are two int32 arrays corresponding to q and a
    in which:
        array[i]==1 when the word is a overlap word in both q and a
        array[i]==0 if the word is not an overlap
        array[i]==2 if it is an empty word (padding)

    Params:
        questions:    string(num_samples, q_len<max_sent_length)
        answers:    string(num_samples, a_len<max_sent_length)
        stoplist:    string(list_len,)    list of stop words
        max_q_length:    int    max sentense length of questions
        max_a_length:    int    max sentense length of answers

    Returns:
        q_overlap:    int(num_samples, max_sent_length)
        a_overlap:    int(num_samples, max_sent_length)
    '''
    stoplist = stoplist if stoplist else []
    feats_overlap = []
    q_indices, a_indices = [], []
    for question, answer in zip(questions, answers):
        q_set = set([q for q in question if q not in stoplist])
        a_set = set([a for a in answer if a not in stoplist])
        word_overlap = q_set.intersection(a_set)
        # so 0 is non-overlap, 1 is overlap, and 2 is empty word
        q_idx = numpy.ones(max_q_length) * 2        
        for i, q in enumerate(question):
            value = 0
            if q in word_overlap:
                value = 1
            q_idx[i] = value
        q_indices.append(q_idx)
        a_idx = numpy.ones(max_a_length) * 2
        for i, a in enumerate(answer):
            value = 0
            if a in word_overlap:
                value = 1
            a_idx[i] = value
        a_indices.append(a_idx)
    q_indices = numpy.vstack(q_indices).astype('int32')
    a_indices = numpy.vstack(a_indices).astype('int32')
    return q_indices, a_indices


def __compute_idf_given_word_vector(docs):
    '''
    Compute Inverse Document Frequency (IDF) given q, a as word vectors

    IDF=log(N/len(doc if w in doc))

    Params:
        docs:   string(n_docs,)    documents

    Returns:
        {word:idf}
    '''
    word2df = defaultdict(float)
    for doc in docs:
        for w in set(doc):
            word2df[w] += 1.0
    num_docs = len(docs)
    for w, value in word2df.iteritems():
        word2df[w] = numpy.math.log(num_docs / value)
    return word2df


def __convert2indices(data, alphabet, 
                    unknown_word_idx_inc=0,
                    empty_word_idx_inc=1, 
                    max_sent_length=MAX_SENT_LENGTH):
    '''
    Convert a list of questions/answers into a indexes, given a dictionary of 
    words.

    Words out of bound will be represented with len(alphabet)+unknown_word_idx_inc
    Results are padded to max_sent_length, in which empty words are represented 
    with len(alphabet)+empty_word_idx_inc

    Params:
        data:    string(n_samples, len_of_doc)    list of questions/answers
        alphabet:    {string:int}
        unknown_word_idx_inc:    int    how to represent unknown word
        empty_word_idx_inc:    int     how to represent empty word
        max_sent_length:    int    max sentense length

    Returns:
        numpy.array(n_samples, max_sent_length)
    '''
    data_idx = []
    unknown_word_idx = len(alphabet) + unknown_word_idx_inc
    empty_word_idx = len(alphabet) + empty_word_idx_inc
    num_unknown_words = 0
    for sentence in data:
        ex = numpy.ones(max_sent_length) * empty_word_idx
        for i, token in enumerate(sentence):
            idx = alphabet.get(token, unknown_word_idx)
            if (idx == unknown_word_idx):
                num_unknown_words += 1
            ex[i] = idx
        data_idx.append(ex)
    data_idx = numpy.array(data_idx, dtype='int32')
    logging.info('Number of unknown words: {}'.format(num_unknown_words))
    return data_idx


class NegativeSamplingIterator(object):
    """ select a positive example and (batch-size - 1) random negative ones """
    def __init__(self, rng, qadataset, batch_size=100):
        self.rng = rng
        self.batch_size = batch_size
        self.n_samples = qadataset.qids.shape[0]
        self.dataset = qadataset
        self.n_batches = (self.n_samples + self.batch_size - 1) / self.batch_size
        self.positive_example_idx = [idx for idx in xrange(self.n_samples) 
                                     if self.dataset.labels[idx] == 1]
        self.n_positive = len(self.positive_example_idx)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        n_batches = self.n_batches
        batch_size = self.batch_size
        n_samples = self.n_samples
        for _ in xrange(n_batches):
            i = self.rng.randint(self.n_positive)
            pos = self.positive_example_idx[i]
            qid_pos = self.dataset.qids[pos]

            q = [self.dataset.q[pos]]
            a = [self.dataset.a[pos]]
            q_overlap = [self.dataset.q_overlap[pos]]
            a_overlap = [self.dataset.a_overlap[pos]]
            labels = [self.dataset.labels[pos]]
            q_pos = q[0]
            q_overlap_pos = q_overlap[0]
            # look for negative examples
            while len(labels) < self.batch_size:
                i = self.rng.randint(self.n_samples)
                if (self.dataset.qids[i] != qid_pos):
                    q.append(q_pos)
                    q_overlap.append(q_overlap_pos)
                    a.append(self.dataset.a[i])                    
                    a_overlap.append(self.dataset.a_overlap[i])
                    labels.append(0)
            yield [numpy.array(q, dtype=numpy.float32), 
                         numpy.array(a, dtype=numpy.float32), 
                         numpy.array(q_overlap, dtype=numpy.float32),
                         numpy.array(a_overlap, dtype=numpy.float32),
                         numpy.array(labels, dtype=numpy.float32)]


def load_dataset(inputdir, alphabet,
                 with_overlap_features = False,
                 stoplist = None):
    '''
    Load a raw dataset and convert to indexes with the embedding.

    Params:
        inputdir:    string    path to the directory where raw data is stored
        alphabet:    {string:int}    word to index
        with_overlap_features:    bool    whether to include overlap features
        stoplist:    string[]    list of stop words when computing overlap features

    Returns:
        QADataset
    '''
    raw_data = __load_raw_dataset(inputdir)
    logging.info('converting q into indices')
    max_q_length = max([len(q) for q in raw_data.q])
    q = __convert2indices(raw_data.q, alphabet, max_sent_length=max_q_length)
    logging.info('converting a into indices')
    max_a_length = max([len(a) for a in raw_data.a])
    a = __convert2indices(raw_data.a, alphabet, max_sent_length=max_a_length)
    if (with_overlap_features):
        q_overlap, a_overlap = __compute_overlap_idx_given_word_vector(raw_data.q, 
                                                                       raw_data.a,
                                                                       stoplist,
                                                                       max_q_length,
                                                                       max_a_length)
    else:
        q_overlap = None
        a_overlap = None
    return QADataset(raw_data.qids,
                     q,
                     a,
                     numpy.asarray(raw_data.labels),
                     q_overlap,
                     a_overlap)


def extract_word_vectors(indices, vectors):
    '''
    Extract word vectors given indices

    Params:
        indices:    int[num_samples, sentense_length]    indexes of each word
        vectors:    numpy.array(num_embedding_vectors, embedding_dim)

    Returns:
        float[num_samples, sentense_length, embedding_dim]
    '''
    return vectors[indices.astype(numpy.int).flatten()].reshape(
        (indices.shape[0], indices.shape[1], vectors.shape[1]))


if (__name__ == '__main__'):
    '''
    Parses the input and dump as pickle files
    '''
    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input tsv/xml file', 
                        required=True)
    parser.add_argument('-o', '--output', help='output directory', 
                        required=True)
    parser.add_argument('-m', '--format', type=str, default='trec', 
                        choices=['trec', 'tsv'], 
                        help='format of the file')
    parser.add_argument('-d', '--debug', default=False, action='store_true', 
                        help='enable ptvsd debugging')
    args = parser.parse_args()

    if (args.debug):
        enable_ptvsd()
    
    __convert_dataset(args.input, args.output, args.format)