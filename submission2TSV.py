import alphabet
import cPickle
import os
import sys
import numpy as np
import pandas as pd

def load_tsv(fname):
  questions, answers, urls, labels = [], [], [], []
  for line in open(fname):
    # Query   Url     PassageID       Passage Rating1 Rating2
    qupprr = line.split('\t')
    questions.append(qupprr[0])
    urls.append(qupprr[1])
    answers.append(qupprr[3])
    labels.append(qupprr[5])
  return questions, answers, urls, labels

def main(argv):
  if (len(argv) != 4):
    print('usage: submission2TSV submissionFile originalTsv resultFile')
    exit(1)

  submission_file = argv[1]
  orig_tsv = argv[2]
  result_file = argv[3]

  # load original tsv file
  print('loading original tsv')
  questions, answers, urls, labels = load_tsv(orig_tsv)
  # load submission file
  print('loading submission file')
  submission = pd.read_csv(submission_file, 
                           names=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'],
                           sep=' ')
  assert len(questions) == len(submission)

  # dump result file
  print('dumping result file to {}'.format(result_file))
  rfile = open(result_file, 'w')
  rfile.write('Query\tUrl\tAnswer\n')
  best_row = 0
  best_score = 0
  last_qid = None
  for i, r in enumerate(submission.iterrows()):
    qid = r['qid']
    score = r['sim']
    if (not last_qid):
      last_qid = r['qid']
    if (qid != last_qid):
      # a new query, so let's dump last
      rfile.write('{}\t{}\t{}\n'.format(questions[best_row], urls[best_row], answers[best_row]))
      best_row = i
      best_score = score
    else:
      if (best_score < score):
        best_score = score
        best_row = i
    last_qid = qid
  rfile.close()

  # dump file for analysis
  print('dumping file for analysis')
  result = pd.DataFrame(columns=['label', 'score', 'q', 'a'])
  result['label'] = labels
  result['score'] = submission['sim']
  result['q'] = questions
  result['a'] = answers
  result.to_csv(result_file, sep='\t')  

if __name__ == '__main__':
  main(sys.argv)