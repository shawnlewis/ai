#!/bin/sh

PYTHONPATH=~/tp_code/models/slim
export PYTHONPATH

DATASET_DIR=~/datasets/kaggle-sealions/tensorflow
TRAIN_DIR=~/train/kaggle-sealions/current
if [ -n "$2" ]; then
    checkpoint_path="$2"
else
    checkpoint_path="$TRAIN_DIR"
fi
python eval_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=sealions \
    --dataset_split_name=$1 \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=${checkpoint_path}
