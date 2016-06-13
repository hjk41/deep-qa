#!/bin/bash

## This file shows how to run training

### First, we need to parse the raw data into consumable format
## We need three data: train, validation and test.
## XML and TSV files are both supported.
train_file=jacana-qa-naacl2013-data-results/train.xml
validation_file=jacana-qa-naacl2013-data-results/dev.xml
test_file=jacana-qa-naacl2013-data-results/test.xml

train_dir=parseddata/50dim/trec/train
validation_dir=parseddata/50dim/trec/validation
test_dir=parseddata/50dim/trec/test

## We need a embedding file, processed with parse_embeddings.py
embedding=embeddings/word2vec/50dim
## Each file will be parsed and will generate several files, so we need
## to specify a directory to store the files for each input
python parse.py -i $train_file -o $train_dir -e $embedding
python parse.py -i $validation_file -o $validation_dir -e $embedding
python parse.py -i $test_file -o $test_dir -e $embedding

### Now we can run training and testing, we can also do both at
##  the same time by specifying 'all' as the first parameter to
##  run_nnet.py.

## output file will store the result for each test sample
output_file=exp.out/50dim/trec/output.txt
## model file will store the model parameters
model_file=exp.out/50dim/trec/model

python run_nnet.py train --train $train_dir --validation $validation_dir --embed $embedding --model $model_file
python run_nnet.py test --test $test_dir --output $output_file --embed $embedding --model $model_file
