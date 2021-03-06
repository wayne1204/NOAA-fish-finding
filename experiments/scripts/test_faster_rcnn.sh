#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in 
  mouss_seq0)
    TRAIN_IMDB="mouss_seq0_trainval"
    TEST_IMDB="mouss_seq0_test"
    STEPSIZE="[50000]"
    ITERS=30000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  mouss_seq1)
    TRAIN_IMDB="mouss_seq1_trainval"
    TEST_IMDB="mouss_seq1_test"
    STEPSIZE="[50000]"
    ITERS=30000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  mbari_seq0)
    TRAIN_IMDB="mbari_seq0_trainval"
    TEST_IMDB="mbari_seq0_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  habcam_seq0)
    TRAIN_IMDB="habcam_seq0_train"
    TEST_IMDB="habcam_seq0_test"
    STEPSIZE="[50000]"
    ITERS=200000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
fi

