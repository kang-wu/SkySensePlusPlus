#!/bin/bash

cd finetune/
cfg=$1
gpu_num=8

bash finetune/tools/dist_train.sh $cfg $gpu_num


