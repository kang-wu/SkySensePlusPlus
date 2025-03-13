# SkySense++ 
This repository is the official implementation of the paper "SkySense++: A Semantic-Enhanced Multi-Modal Remote Sensing Foundation Model for Earth Observation".

## 📢 Latest Updates
🔥🔥🔥 Last Updated on 2024.3.12 🔥🔥🔥
- [2025.3.12] updated sentinel-1 images and labels of JL-16 dataset 
- [2025.3.9] updated pretrain and evaluation data
- [2024.11.13]updated pretrain and evaluation data

## Pretrain Data
### RS-Semantic Dataset
We conduct semantic-enhanced pretraining on the RS-Semantic dataset, which consists of 13 datasets with pixel-level annotations. Below are the specifics of these datasets. (also see in [Zenodo](https://zenodo.org/records/15010418).
| Dataset                          | Modalities       | GSD(m) | Size                  | Categories          | Download Link                                                                 |
|----------------------------------|------------------|--------|-----------------------|---------------------|-------------------------------------------------------------------------------|
| Five Billion Pixels              | Gaofen-2         | 4      | 6800x7200             | 24                  | [Download](https://x-ytong.github.io/project/Five-Billion-Pixels.html)                   |
| Potsdam                          | Airborne         | 0.05   | 6000x6000             | 5                   | [Download](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)                               |
| Vaihingen                        | Airborne         | 0.05   | 2494x2064             | 5                   | [Download](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)                            |
| Deepglobe                        | WorldView        | 0.5    | 2448x2448             | 6                   | [Download](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)                            |
| iSAID                            | Multiple Sensors | -      | 800x800 to 4000x13000 | 15                  | [Download](https://captain-whu.github.io/iSAID/index.html)                                 |
| LoveDA                           | Spaceborne       | 0.3    | 1024x1024             | 7                   | [Download](https://github.com/Junjue-Wang/LoveDA)                              |
| DynamicEarthNet                  | WorldView        | 0.3    | 1024x1024             | 7                   | [Download](https://github.com/aysim/dynnet)                     |
|                                  | Sentinel-2*      | 10     | 32x32                 |                     |                                                                               |
|                                  | Sentinel-1*      | 10     | 32x33                 |                     |                                                                               |
| Pastis-MM                        | WorldView        | 0.3    | 1024x1024             | 18                  | [Download](https://github.com/VSainteuf/pastis-benchmark)                           |
|                                  | Sentinel-2*      | 10     | 32x32                 |                     |                                                                               |
|                                  | Sentinel-1*      | 10     | 32x33                 |                     |                                                                               |
| C2Seg-AB                         | Sentinel-2*      | 10     | 128x128               | 13                  | [Download](https://github.com/danfenghong/RSE_Cross-city)                              |
|                                  | Sentinel-1*      | 10     | 128x128               |                     |                                                                               |
| FLAIR                            | Spot-5           | 0.2    | 512x512               | 12                  | [Download](https://github.com/IGNF/FLAIR-2)                          |
|                                  | Sentinel-2*      | 10     | 40x40                 |                     |                                                                               |
| DFC20                            | Sentinel-2       | 10     | 256x256               | 9                   | [Download](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest)                            |
|                                  | Sentinel-1       | 10     | 256x256               |                     |                                                                               |
| S2-naip                          | NAIP             | 1      | 512x512               | 32                  | [Download](https://huggingface.co/datasets/allenai/s2-naip)                               |
|                                  | Sentinel-2*      | 10     | 64x64                 |                     |                                                                               |
|                                  | Sentinel-1*      | 10     | 64x64                 |                     |                                                                               |
| JL-16                            | Jilin-1          | 0.72   | 512x512               | 16                  | [Download](https://zenodo.org/records/15010418)                                 |
|                                  | Sentinel-1*      | 10     | 40x40                 |                     |                                                                               |

*\* for time-series data.*
## EO Benchmark
We evaluate our SkySense++ on 12 typical Earth Observation (EO) tasks across 7 domains: *agriculture*, *forestry*, *oceanography*, *atmosphere*, *biology*, *land surveying*, and *disaster management*. The detailed information about the datasets used for evaluation is as follows.
| Domain                          | Task type                  | Dataset                          | Modalities                                                                 | GSD  | Image size          | Download Link                                                                 | Notes                                                                 |
|---------------------------------|----------------------------|----------------------------------|----------------------------------------------------------------------------|------|---------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Agriculture                     | Crop classification        | Germany                          | Sentinel-2*                                                                | 10   | 24x24               | [Download](https://github.com/michaeltrs/DeepSatModels/tree/main/data)                               |                                                                       |
| Foresetry                       | Tree species classification | TreeSatAI-Time-Series            | Airborne,                                          | 0.2  | 304x304             | [Download](http://example.com/download/treesatai-time-series)                |                                                                       |
|                                 |                            |                                  | Sentinel-2*                                                                | 10   | 6x6                 |         |      |                                                                       |
|                                 |                            |                                  | Sentinel-1*                                                                | 10   | 6x6                 |         |                                                                       |
|                                 | Deforestation segmentation | Atlantic                         | Sentinel-2                                                                 | 10   | 512x512             | [Download](https://github.com/davej23/attention-mechanism-unet)                    |                                                                       |
| Oceanography                    | Oil spill segmentation     | SOS                              | Sentinel-1                                                                 | 10   | 256x256             | [Download](https://grzy.cug.edu.cn/zhuqiqi/en/yjgk/32384/list/index.htm)                                 |                                                                       |
| Atmosphere                      | Air pollution regression   | 3pollution                       | Sentinel-2                                             | 10   | 200x200             | [Download](https://github.com/CoDIS-Lab/AQNet)                           |                                                                       |
|                                 |                            |                                  | Sentinel-5P                                                                | 2600 | 120x120             |               |                                                                       |
| Biology                         | Wildlife detection         | Kenya                            | Airborne                                                                   | -    | 3068x4603           | [Download](https://data.4tu.nl/articles/_/12713903/1)                                |                                                                       |
| Land surveying                  | LULC mapping               | C2Seg-BW                         | Gaofen-6                                                      | 10   | 256x256             | [Download](https://github.com/danfenghong/RSE_Cross-city)                             |                                                                       |
|                                 |                            |                                  | Gaofen-3                                                                   | 10   | 256x256             |                   |                                                                       |
|                                 | Change detection           | dsifn-cd                         | GoogleEarth                                                                | 0.3  | 512x512             | [Download](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)                      |                                                                       |
| Disaster management             | Flood monitoring           | Flood-3i                         | Airborne                                                                   | 0.05 | 256 × 256           | [Download](https://drive.google.com/drive/folders/1FMAKf2sszoFKjq0UrUmSLnJDbwQSpfxR)                           |                                                                       |
|                                 |                            | C2SMSFloods                      | Sentinel-2, Sentinel-1                                                     | 10   | 512x512             | [Download](https://beta.source.coop/c2sms/)                         |                                                                       |
|                                 | Wildfire monitoring        | CABUAR                           | Sentinel-2                                                                 | 10   | 5490 × 5490         | [Download](https://github.com/DarthReca/CaBuAr)                               |                                                                       |
|                                 | Landslide mapping          | GVLM                             | GoogleEarth                                                                | 0.3  | 1748x1748 ~ 10808x7424 | [Download](https://github.com/zxk688/GVLM)                                 |                                                                       |
|                                 | Building damage assessment | xBD                              | WorldView                                                                  | 0.3  | 1024x1024           | [Download](https://xview2.org/)                                   |                                                                       |

*\* for time-series data.*

## Code
We will release the code and pretraining weights after company's open-source approval.

## Acknowledgments
- [antmmf](https://github.com/alipay/Ant-Multi-Modal-Framework)
- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [Painter](https://github.com/baaivision/Painter)

