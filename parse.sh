#!/bin/bash

python parse.py jacana-qa-naacl2013-data-results/train.xml parseddata/trec/train embeddings/word2vec
python parse.py jacana-qa-naacl2013-data-results/dev.xml parseddata/trec/dev embeddings/word2vec
python parse.py jacana-qa-naacl2013-data-results/test.xml parseddata/trec/test embeddings/word2vec
