#!/bin/bash
# bash ./scripts-search/train-models.sh 0/1 0 100 -1 '777 888 999'
echo script name: $0
echo $# arguments
if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for use-less-or-not, start-and-end, arch-index, and seeds"
  exit 1
fi
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
	--datasets cifar10 cifar10 cifar100 ImageNet16-120 \
	--splits   1       0       0        0 \
	--xpaths $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python/ImageNet16 \
	--workers 8