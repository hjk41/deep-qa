#!/usr/bin/python

'''
Utility to load and convert datasets
'''

import numpy
import pickle

class QADataset():
    def __init__(self, qids, q, a, q_overlap, a_overlap, labels):
        self.qids = qids
        self.q = q
        self.a = a
        self.q_overlap = q_overlap
        self.a_overlap = a_overlap
        self.labels = labels

MAX_SENT_LENGTH = 70

def load_trec(fname, max_sent_length=MAX_SENT_LENGTH):
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
    print('num_skipped: {}'.format(num_skipped))
    return question2qid.keys(), qids, questions, answers, labels, []