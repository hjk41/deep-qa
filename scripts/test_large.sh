#!/bin/bash

## This file shows how to run training

### First, we need to parse the raw data into consumable format
## We need three data: train, validation and test.
## XML and TSV files are both supported.
test_file=~/qa/data/relevance/superdotest_converted.tsv
test_dir=parseddata/relevance/superdo

## We need a embedding file, processed with parse_embeddings.py
embedding=embeddings/word2vec
## Each file will be parsed and will generate several files, so we need
## to specify a directory to store the files for each input
python parse.py -i $test_file -o $test_dir -e $embedding -g positive

### Now we can run training and testing, we can also do both at
##  the same time by specifying 'all' as the first parameter to
##  run_nnet.py.

## output file will store the result for each test sample
output_file=exp.out/relevance/output.txt
## model file will store the model parameters
model_file=exp.out/relevance/model

python run_nnet.py test --test $test_dir --output $output_file --embed $embedding --model $model_file

### Convert the output to readable format
## The output consists of a lot of ids, which is for trec_eval.
## The submssion2TSV.py script will convert it to include original
## query and answer texts so we can analyze it.
readable_result_file=exp.out/relevance/superdo.tsv
python submission2TSV.py $output_file $test_file $readable_result_file
