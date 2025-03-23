# SkySense++

This repository is the official implementation of the paper "SkySense++: A Semantic-Enhanced Multi-Modal Remote Sensing Foundation Model Beyond SkySense for Earth Observation".

## 📢 Latest Updates

🔥🔥🔥 Last Updated on 2025.03.23 🔥🔥🔥

- [2025.03.23] We have obtained the approval by the open-source committee of the Company. We have updated code for data preprocessing, pretraining, and application.
- [2025.03.14] updated optical images of JL-16 dataset in [Huggingface](https://huggingface.co/datasets/KKKKKKang/JL-16).
- [2025.03.12] updated sentinel-1 images and labels of JL-16 dataset  in [Zenodo](https://zenodo.org/records/15010418).
- [2025.03.09] created repo in [Zenodo](https://zenodo.org/records/15010418), datasets are uploading.
- [2024.11.13] updated details of pretrain and evaluation data.

## Pretrain Data

### RS-Semantic Dataset

We conduct semantic-enhanced pretraining on the RS-Semantic dataset, which consists of 13 datasets with pixel-level annotations. Below are the specifics of these datasets. (also see in [Zenodo](https://zenodo.org/records/15010418)).

| Dataset             | Modalities       | GSD(m) | Size                  | Categories | Download Link                                                                                 |
| ------------------- | ---------------- | ------ | --------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| Five Billion Pixels | Gaofen-2         | 4      | 6800x7200             | 24         | [Download](https://x-ytong.github.io/project/Five-Billion-Pixels.html)                           |
| Potsdam             | Airborne         | 0.05   | 6000x6000             | 5          | [Download](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)     |
| Vaihingen           | Airborne         | 0.05   | 2494x2064             | 5          | [Download](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)   |
| Deepglobe           | WorldView        | 0.5    | 2448x2448             | 6          | [Download](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) |
| iSAID               | Multiple Sensors | -      | 800x800 to 4000x13000 | 15         | [Download](https://captain-whu.github.io/iSAID/index.html)                                       |
| LoveDA              | Spaceborne       | 0.3    | 1024x1024             | 7          | [Download](https://github.com/Junjue-Wang/LoveDA)                                                |
| DynamicEarthNet     | WorldView        | 0.3    | 1024x1024             | 7          | [Download](https://github.com/aysim/dynnet)                                                      |
|                     | Sentinel-2*      | 10     | 32x32                 |            |                                                                                               |
|                     | Sentinel-1*      | 10     | 32x33                 |            |                                                                                               |
| Pastis-MM           | WorldView        | 0.3    | 1024x1024             | 18         | [Download](https://github.com/VSainteuf/pastis-benchmark)                                        |
|                     | Sentinel-2*      | 10     | 32x32                 |            |                                                                                               |
|                     | Sentinel-1*      | 10     | 32x33                 |            |                                                                                               |
| C2Seg-AB            | Sentinel-2*      | 10     | 128x128               | 13         | [Download](https://github.com/danfenghong/RSE_Cross-city)                                        |
|                     | Sentinel-1*      | 10     | 128x128               |            |                                                                                               |
| FLAIR               | Spot-5           | 0.2    | 512x512               | 12         | [Download](https://github.com/IGNF/FLAIR-2)                                                      |
|                     | Sentinel-2*      | 10     | 40x40                 |            |                                                                                               |
| DFC20               | Sentinel-2       | 10     | 256x256               | 9          | [Download](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest)            |
|                     | Sentinel-1       | 10     | 256x256               |            |                                                                                               |
| S2-naip             | NAIP             | 1      | 512x512               | 32         | [Download](https://huggingface.co/datasets/allenai/s2-naip)                                      |
|                     | Sentinel-2*      | 10     | 64x64                 |            |                                                                                               |
|                     | Sentinel-1*      | 10     | 64x64                 |            |                                                                                               |
| JL-16               | Jilin-1          | 0.72   | 512x512               | 16         | [Download](https://zenodo.org/records/15010418)                                                  |
|                     | Sentinel-1*      | 10     | 40x40                 |            |                                                                                               |

*\* for time-series data.*

### RS-Representation Dataset

The pretraining list is in the [Zenodo](https://zenodo.org/records/15068572)- `rep_data_list.tar`. The download and process scripts are in [tools/pretraining_data_builder](tools/pretraining_data_builder).

## EO Benchmark

We evaluate our SkySense++ on 12 typical Earth Observation (EO) tasks across 7 domains: *agriculture*, *forestry*, *oceanography*, *atmosphere*, *biology*, *land surveying*, and *disaster management*. The detailed information about the datasets used for evaluation is as follows.

| Domain              | Task type                   | Dataset               | Modalities             | GSD  | Image size             | Download Link                                                                                                                                 | Notes |
| ------------------- | --------------------------- | --------------------- | ---------------------- | ---- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| Agriculture         | Crop classification         | Germany               | Sentinel-2*            | 10   | 24x24                  | [Download](https://github.com/michaeltrs/DeepSatModels/tree/main/data)                                                                           |       |
| Foresetry           | Tree species classification | TreeSatAI-Time-Series | Airborne,              | 0.2  | 304x304                | [Download](http://example.com/download/treesatai-time-series)                                                                                    |       |
|                     |                             |                       | Sentinel-2*            | 10   | 6x6                    |                                                                                                                                               |       |
|                     |                             |                       | Sentinel-1*            | 10   | 6x6                    |                                                                                                                                               |       |
|                     | Deforestation segmentation  | Atlantic              | Sentinel-2             | 10   | 512x512                | [Download](https://github.com/davej23/attention-mechanism-unet)                                                                                  |       |
| Oceanography        | Oil spill segmentation      | SOS                   | Sentinel-1             | 10   | 256x256                | [Download](https://grzy.cug.edu.cn/zhuqiqi/en/yjgk/32384/list/index.htm)                                                                         |       |
| Atmosphere          | Air pollution regression    | 3pollution            | Sentinel-2             | 10   | 200x200                | [Download](https://github.com/CoDIS-Lab/AQNet)                                                                                                   |       |
|                     |                             |                       | Sentinel-5P            | 2600 | 120x120                |                                                                                                                                               |       |
| Biology             | Wildlife detection          | Kenya                 | Airborne               | -    | 3068x4603              | [Download](https://data.4tu.nl/articles/_/12713903/1)                                                                                            |       |
| Land surveying      | LULC mapping                | C2Seg-BW              | Gaofen-6               | 10   | 256x256                | [Download](https://github.com/danfenghong/RSE_Cross-city)                                                                                        |       |
|                     |                             |                       | Gaofen-3               | 10   | 256x256                |                                                                                                                                               |       |
|                     | Change detection            | dsifn-cd              | GoogleEarth            | 0.3  | 512x512                | [Download](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset) |       |
| Disaster management | Flood monitoring            | Flood-3i              | Airborne               | 0.05 | 256 × 256             | [Download](https://drive.google.com/drive/folders/1FMAKf2sszoFKjq0UrUmSLnJDbwQSpfxR)                                                             |       |
|                     |                             | C2SMSFloods           | Sentinel-2, Sentinel-1 | 10   | 512x512                | [Download](https://beta.source.coop/c2sms/)                                                                                                      |       |
|                     | Wildfire monitoring         | CABUAR                | Sentinel-2             | 10   | 5490 × 5490           | [Download](https://github.com/DarthReca/CaBuAr)                                                                                                  |       |
|                     | Landslide mapping           | GVLM                  | GoogleEarth            | 0.3  | 1748x1748 ~ 10808x7424 | [Download](https://github.com/zxk688/GVLM)                                                                                                       |       |
|                     | Building damage assessment  | xBD                   | WorldView              | 0.3  | 1024x1024              | [Download](https://xview2.org/)                                                                                                                  |       |

*\* for time-series data.*

## Implementation Code

### Structure

This project mainly contains the following parts.

```plain
./
├── antmmf/                             # antmmf framework code
├── configs/                   
│   ├── eval_skysense_pp_flood3i.yml    # eval cfg on flood3i                
│   └── pretrain_skysensepp.yml         # pretrain cfg
├── finetune/                           # finetuning code
│   ├── configs/                        # finetuning configs
│   ├── mmseg/                          # mmseg library
│   ├── requirements/                   # mmseg install requirements folder
│   ├── requirements.txt                # mmseg install requirements
│   ├── setup.py                        # mmseg setup file
│   └── tools/                          # mmseg utils
├── lib/                                # model implementation
│   ├── datasets/                       # datasets for evaluation
│   ├── evaluation/                     # evaluation code
│   ├── models/                         # model architecture
│   ├── predictors/                     # inference code
│   ├── task/                           # task code
│   ├── trainer/                        # trainer code
│   ├── utils/                          # library code
│   └── __init__.py                     # packages init file
├── pretrain/                           # pretrain ckpts
├── tools/                              # tools ckpts
│   ├── pretraining_data_builder        # pretraining dataset builder
│   ├── run_1shot_flood3i.sh            # datasets for evaluation
│   ├── run_ft_atlantic.sh              # run ft script
│   ├── run_pretrain.sh                 # run pretrain script
│   └── run.py                          # Program entry point
└── readme.md                           # project readme
```

### Environment

Each machine for implementating the pretraining or fintuning are with *Alibaba Group Enterprise Linux(7.2)* and *Python 3.8.10*. The pretraining and finetuning code are implemented on severs with *Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz* and *Nvidia A100 GPUS*.

### Pretraining

To run our pretraining code, please install dependency packages. (Instalazation takes about 14 minutes on a node with Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz and 8 A100 GPUs.)

```plain
torch==1.13.1
atorch==0.1.3
torchvision==0.14.1
mmcv-full==2.1.0
mmsegmentation==1.2.2
timm==0.6.13
gdal==3.4.0
scikit-image==0.19.3
```

Step1. Install the above packages and clone [antmmf framework](https://github.com/alipay/Ant-Multi-Modal-Framework):

```bash
git clone https://github.com/alipay/Ant-Multi-Modal-Framework.git antmmf/
```

Step2. Download the pretraining datasets in [Zenodo](https://zenodo.org/records/15010418) and orgnize them as follows:

```
pretrain_datasets
├── dynamic-mm                          # multi-modal dynamic-mm datasets
│   ├── images_hr                       # hr images
│   ├── images_s2                       # sentinel-2 images
│   ├── images_s1                       # sentinel-1 images
│   ├── labels                          # segmentation annotations
│   ├── dynamic-mm_train.json           # train list file
│   └── dynamic-mm_val.json             # val list file
├── fbp                                 # single-modal fbp datasets
│   ├── images                          # input gaofen-2 images
│   ├── labels                          # segmentation annotations
│   ├── fbp_train.json                  # train list file
│   └── fbp_val.json                    # val list file
└── ......                       
```

The `<dataset>_<train/val>.json` is used to read information for training and validation, with a unified organizational format:

```json
[
  {
    "hr_path": "dataset_name/images_hr/<img_name>.png", // hr info c,h,w
    "s2_path": ["dataset_name/images_s2/<img_name>_20240101.npz", "dataset_name/images_s2/<img_name>_20240103.npz"], // s2 c,h,w
    "s1_path": ["dataset_name/images_s1/<img_name>_20240104.npz", "dataset_name/images_s1/<img_name>_20240108.npz"], // s1 c,h,w
    "target_path": "dataset_name/labels/<img_name>.png", // annotation info
    "type": "dataset_name", // dataset_name
    "classes": [            // Included categories
            0,
            2,
            4,
            5
        ]
  },
  {
    ...
  }
]
```

Step3. Download the pretraining weights of SkySense [here](https://www.yuque.com/thinson/research/vpisiuswzbnriwvb?singleDoc=&language=en-us) (access code:bkl3) and move it to `pretrain/`

Step4. Run the pretrain code on 4 nodes (each node with 8 A100 gpus):

```
bash tools/run_pretrain.sh <node_rank:0-3> <master_ip_address>
```

For example, if the ip adress of master node is *192.168.112.10*, the command for node *1* is:

```
bash tools/run_pretrain.sh 1 192.168.112.10
```

### Downstream 1-shot application

#### Requirments

To run our code, please install dependency packages. ( Instalazation takes about 10 minutes on a sever with Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz and 2 A100 GPUs.)

```plain
torch==1.13.1
atorch==0.1.3
torchvision==0.14.1
mmcv-full==1.7.1
mmcls==0.25.0
mmsegmentation==0.30.0
timm==0.6.13
gdal==3.4.0
scikit-image==0.19.3
```

#### Run steps

step1. Clone [antmmf framework](https://github.com/alipay/Ant-Multi-Modal-Framework). and install the above packages:

```plain
git clone https://github.com/alipay/Ant-Multi-Modal-Framework.git antmmf/
```

step1. Download the flood-3i dataset at [here](https://drive.google.com/drive/folders/1FMAKf2sszoFKjq0UrUmSLnJDbwQSpfxR).

step2. Using the above pretraining wieights or download the pretrained model weights [here](https://www.yuque.com/thinson/research/vpisiuswzbnriwvb?singleDoc=&language=en-us) (access code:bkl3).

step3. Run the script for evaluating 1-shot performance on flood-3i:

```plain
bash tools/run_1shot.sh <gpu_idx> flood-3i(dataset_name)
```

### Downstream finetuning application

#### Requirments

We build our fine-tuning application code on the [openmmlab framework](https://github.com/open-mmlab/mmcv).

To run our code, please install dependency packages. ( Instalazation takes about 10 minutes on a sever with Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz and 2 A100 GPUs.)

```plain
torch==1.13.1
torchvision==0.14.1
mmcv-full==2.1.0
mmpretrain==1.2.0
mmsegmentation==1.2.2
mmdetection==3.3.0
timm==0.6.13
gdal==3.4.0
scikit-image==0.19.3
```

#### Run steps

Step1. Install the mmsegmentation framework under the instrction in [here](https://mmsegmentation.readthedocs.io/en/latest/index.html)

Step2. Download the evaluation datsets. We take  Atlantic dataset for deforestation segmentation as an example. Download the Atlantic dataset  at [here](https://github.com/davej23/attention-mechanism-unet).

Step3. Use your pretrained model weights or download the model weights: [here](https://www.yuque.com/thinson/research/vpisiuswzbnriwvb?singleDoc=&language=en-us) (access code:bkl3)

Step4. Run the finetuning script. We take the Atlantic dataset as an example:

```bash
bash tools/run_finetune.sh configs/atlantic.py
```

## Acknowledgments

This projects are mainly built on the following projects:

+ [antmmf](https://github.com/alipay/Ant-Multi-Modal-Framework)
+ [mmcv](https://github.com/open-mmlab/mmcv)
+ [mmpretrain](https://github.com/open-mmlab/mmpretrain)
+ [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
+ [mmdetection](https://github.com/open-mmlab/mmdetection)
+ [Painter](https://github.com/baaivision/Painter)

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) and is intended solely for non-commercial research purposes.
