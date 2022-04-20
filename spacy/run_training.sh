#!/bin/bash
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
cuda="$1"

if [[ $# -ne 1 ]] ; then
  echo 'Usage: run_traing.sh <cuda>'
  exit 1
fi

python -m spacy project run train . --vars.gpu "$cuda" --vars.run_name "$TIMESTAMP-cuda-$cuda"
