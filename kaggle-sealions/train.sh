#!/bin/sh

PYTHONPATH=../../models/slim
export PYTHONPATH

DATASET_DIR=dataset/tensorflow
TRAIN_DIR=train
CHECKPOINT_PATH=checkpoints/inception_resnet_v2_2016_08_30.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=sealions \
    --dataset_split_name=train \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --optimizer=adam \
    --learning_rate=.001
