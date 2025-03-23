#!/bin/sh
set -x
export PYTHONPATH=`pwd`:$PYTHONPATH
cd antmmf
export PYTHONPATH=`pwd`:$PYTHONPATH
cd ..
export NCCL_DEBUG=INFO

export OMP_NUM_THREADS=4

NUM_GPU=8

CONFIG_FILE=configs/pretrain_skysensepp.yml
SAVE_DIR=save/skysensepp_pretrain

mkdir -p ${SAVE_DIR}/$1/

pip install lmdb
nohup python -m antmmf.utils.launch --nproc_per_node=${NUM_GPU} --master_port 12345 --nnodes=4 --node_rank=$1 --master_addr=$2 tools/run.py --config $CONFIG_FILE \
    training_parameters.distributed True \
    training_parameters.save_dir ${SAVE_DIR} > ${SAVE_DIR}/$1/nohup.log 2>&1 &

sleep 3s
tail -f ${SAVE_DIR}/$1/nohup.log
