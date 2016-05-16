#!/bin/python
import sys


def main(argv):
  '''
  This script converts superdo data to hb format by 
  adding two columns to the back of each line.

  SuperDo data has format "Query Url PassageID Passage",
  while HB data has format "Query Url PassageID Passage Rating1 Rating2"
  Here we just append Rating 1 as 0, Rating2 as Bad
  '''
  if (len(argv) != 3):
    print('usage: convertSuperDo.py superDo.tsv output.tsv')
    exit(1)
  input = argv[1]
  output = argv[2]
  of = open(output, 'w')
  for i, line in enumerate(open(input)):
    if (i == 0):
      # get rid of tsv header
      continue
    of.write('{}\t0\tBad\n'.format(line.strip()))
  of.close()

if __name__ == '__main__':
  main(sys.argv)