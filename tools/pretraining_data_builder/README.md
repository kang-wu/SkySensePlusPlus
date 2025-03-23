# Pretraining Data Builder
This code is for building pretraining data for the self-supervised learning of SkySense++.

## Install
Prepare the environment:
```
conda create -n data_builder python=3.12
conda activate data_builder
pip install -r requirements.txt
```
Download pretraining data list in lmdb format from [Zenodo](https://zenodo.org/records/14994430)

## Download Data
```
python -m rsi_download --username <username> --password <password> --api_key <api_key> <X> <Y> <Z> <date_min> <date_max>
```
Notes:
1. `username` and `password` can be created in the [Copernicus Data Space Ecosystem](https://data.copernicus.eu/cdsapp/#!/home), 
`api_key` can be created in the [Maxar](https://ard.maxar.com/docs/about/).
2. `X` `Y` `Z` are coordinates in the Web Mercator coordinate system.
3. `date_min` and `date_max` are in the format of `YYYY-MM`.

## Process Data
```
python -m rsi_process --platform <platform> --fn_img path/to/image.zip --save_dir output_<platform>/
```
Notes:
1. `platform` can be `s1`, `s2`, `wv`.
2. `fn_img` is the path to the downloaded zip file.
3. `save_dir` is the directory to save the processed data.

## Automatic Script
```
sh run_data_builder.sh
```
This script will first read the pretraining list, then download the data according to the list, and proceed them automatically.
