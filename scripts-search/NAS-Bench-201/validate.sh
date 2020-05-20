#!/bin/bash
# bash ./scripts-search/train-models.sh 0/1 0 100 -1 '777 888 999'
echo script name: $0
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

arch_path=$1
ckp_path=$2

OMP_NUM_THREADS=4 python ./exps/NAS-Bench-201/validate_super.py \
  --arch_path ${arch_path} \
  --ckp_path ${ckp_path} \
	--datasets cifar10 \
	--splits   1       0       0        0 \
	--xpaths $TORCH_HOME/cifar.python \
	--workers 8
