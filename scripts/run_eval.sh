#!/bin/csh -f

output=$1
judgement=$2

./trec_eval-8.0/trec_eval -q -c ${judgement} ${output} > ${output}.treceval
tail -29 ${output}.treceval | grep -e 'map' -e 'recip_rank'
exit 0
