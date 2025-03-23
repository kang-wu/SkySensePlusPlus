#!/bin/bash
set -x
export PYTHONPATH=`pwd`:$PYTHONPATH
cd antmmf
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ../

export CUDA_VISIBLE_DEVICES=$1
dataset_name=$2

CONFIG_PATH=configs/eval_skysense_pp_${dataset_name}.yml
MODEL_PATH=pretrain/skysensepp_release.ckpt

SAVE_DIR=eval/${dataset_name}_1shot/save
GT_DIR=eval_datasets/${dataset_name}/targets
GT_LIST_PATH=eval_datasets/${dataset_name}/val.txt

mkdir -p $SAVE_DIR

# predictor
python lib/predictors/${dataset_name}_1shot.py \
    --model_path $MODEL_PATH \
    --config $CONFIG_PATH \
    --save_dir $SAVE_DIR \
    --seed 0

# eval
python lib/evaluation/segm_eval_base.py \
   --pred_dir ${SAVE_DIR} \
   --gt_dir ${GT_DIR} \
   --gt_list_path ${GT_LIST_PATH} \
   --gt_suffix '.png' \
   --dataset_name ${dataset_name} \
   --dist_type 'abs' \
   --model_name skysense++_1shot