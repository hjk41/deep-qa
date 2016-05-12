import alphabet
import cPickle
import os
import sys
import numpy as np
import pandas as pd

def idlist2sent(idlist, id2word):
  return ' '.join([id2word[id] for id in idlist])

def main(argv):
  if (len(argv) != 3):
    print('usage: submission2TSV inputDir resultDir')
    exit(1)

  input_dir = argv[1]
  result_dir = argv[2]
  vocab = cPickle.load(open(os.path.join(input_dir, 'vocab.pickle')))
  id2word = {id:word for word, id in vocab.items()}
  id2word[len(vocab)] = ''
  id2word = [id2word[i] for i in range(len(id2word))]
  qids = np.load(os.path.join(input_dir, 'test.qids.npy'))
  questions = np.load(os.path.join(input_dir, 'test.questions.npy'))
  answers = np.load(os.path.join(input_dir, 'test.answers.npy'))
  labels = np.load(os.path.join(input_dir, 'test.labels.npy'))
  submission = pd.read_csv(os.path.join(result_dir, 'submission.txt'), 
                           names=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'],
                           sep=' ')
  qtexts = [idlist2sent(idlist, id2word) for idlist in questions]
  atexts = [idlist2sent(idlist, id2word) for idlist in answers]
  N = len(submission)
  result = pd.DataFrame(columns=['label', 'score', 'q', 'a'])
  result['label'] = labels
  result['score'] = submission['sim']
  result['q'] = qtexts
  result['a'] = atexts
  result.to_csv(os.path.join(result_dir, 'result.tsv'), sep='\t')

if __name__ == '__main__':
  main(sys.argv)