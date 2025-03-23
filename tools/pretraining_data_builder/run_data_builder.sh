#! /bin/bash
source activate data_builder
export USERNAME=your_username
export PASSWORD=your_password
export API_KEY=your_api_key

export PYTHONPATH=$PYTHONPATH:$(pwd)

LMDB_PATH=your_lmdb_path

python rsi_pipeline/data_builder.py $LMDB_PATH