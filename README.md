# SkySense++ 
## Description
This repository is the official implementation of the paper "SkySense++: A Semantic-Enhanced Multi-Modal Remote Sensing Foundation Model for Earth Observation".

## 📢 Latest Updates
🔥🔥🔥 Last Updated on 2024.11.13 🔥🔥🔥
- updated pretrain and evaluation data

## Pretrain Data
### RS-Semantic Dataset
We conduct semantic-enhanced pretraining on the RS-Semantic dataset, which consists of 13 datasets with pixel-level annotations. Below are the specifics of these datasets.
| Dataset                          | Modalities       | GSD(m) | Size                  | Categories          | Download Link                                                                 |
|----------------------------------|------------------|--------|-----------------------|---------------------|-------------------------------------------------------------------------------|
| Five Billion Pixels              | Gaofen-2         | 4      | 6800x7200             | 24                  | [Download](http://example.com/download/five-billion-pixels)                   |
| Potsdam                          | Airborne         | 0.05   | 6000x6000             | 5                   | [Download](http://example.com/download/potsdam)                               |
| Vaihingen                        | Airborne         | 0.05   | 2494x2064             | 5                   | [Download](http://example.com/download/vaihingen)                             |
| Deepglobe                        | WorldView        | 0.5    | 2448x2448             | 6                   | [Download](http://example.com/download/deepglobe)                             |
| iSAID                            | Multiple Sensors | -      | 800x800 to 4000x13000 | 15                  | [Download](http://example.com/download/isaid)                                 |
| LoveDA                           | Spaceborne       | 0.3    | 1024x1024             | 7                   | [Download](http://example.com/download/loveda)                                |
| DynamicEarthNet                  | WorldView        | 0.3    | 1024x1024             | 7                   | [Download](http://example.com/download/dynamicearthnet)                       |
|                                  | Sentinel-2*      | 10     | 32x32                 |                     |                                                                               |
|                                  | Sentinel-1*      | 10     | 32x33                 |                     |                                                                               |
| Pastis-MM                        | WorldView        | 0.3    | 1024x1024             | 18                  | [Download](http://example.com/download/pastis-mm)                             |
|                                  | Sentinel-2*      | 10     | 32x32                 |                     |                                                                               |
|                                  | Sentinel-1*      | 10     | 32x33                 |                     |                                                                               |
| C2Seg-AB                         | Sentinel-2*      | 10     | 128x128               | 13                  | [Download](http://example.com/download/c2seg-ab)                              |
|                                  | Sentinel-1*      | 10     | 128x128               |                     |                                                                               |
| FLAIR                            | Spot-5           | 0.2    | 512x512               | 12                  | [Download](http://example.com/download/flair)                                 |
|                                  | Sentinel-2*      | 10     | 40x40                 |                     |                                                                               |
| DFC20                            | Sentinel-2       | 10     | 256x256               | 9                   | [Download](http://example.com/download/dfc20)                                 |
|                                  | Sentinel-1       | 10     | 256x256               |                     |                                                                               |
| S2-naip                          | NAIP             | 1      | 512x512               | 32                  | [Download](http://example.com/download/s2-naip)                               |
|                                  | Sentinel-2*      | 10     | 64x64                 |                     |                                                                               |
|                                  | Sentinel-1*      | 10     | 64x64                 |                     |                                                                               |
| JL-16                            | Jilin-1          | 0.72   | 512x512               | 16                  | [Download](http://example.com/download/jl-16)                                 |
|                                  | Sentinel-1*      | 10     | 40x40                 |                     |                                                                               |

*\* for time-series data.*
## EO Benchmark
We evaluate our SkySense++ on 12 typical Earth Observation (EO) tasks across 7 domains: *agriculture*, *forestry*, *oceanography*, *atmosphere*, *biology*, *land surveying*, and *disaster management*. The detailed information about the datasets used for evaluation is as follows.
| Domain                          | Task type                  | Dataset                          | Modalities                                                                 | GSD  | Image size          | Download Link                                                                 | Notes                                                                 |
|---------------------------------|----------------------------|----------------------------------|----------------------------------------------------------------------------|------|---------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Agriculture                     | Crop classification        | Germany                          | Sentinel-2*                                                                | 10   | 24x24               | [Download](http://example.com/download/germany)                               |                                                                       |
| Foresetry                       | Tree species classification | TreeSatAI-Time-Series            | Airborne,                                          | 0.2  | 304x304             | [Download](http://example.com/download/treesatai-time-series)                |                                                                       |
|                                 |                            |                                  | Sentinel-2*                                                                | 10   | 6x6                 |         |      |                                                                       |
|                                 |                            |                                  | Sentinel-1*                                                                | 10   | 6x6                 |         |                                                                       |
|                                 | Deforestation segmentation | Atlantic                         | Sentinel-2                                                                 | 10   | 512x512             | [Download](http://example.com/download/atlantic)                             |                                                                       |
| Oceanography                    | Oil spill segmentation     | SOS                              | Sentinel-1                                                                 | 10   | 256x256             | [Download](http://example.com/download/sos)                                   |                                                                       |
| Atmosphere                      | Air pollution regression   | 3pollution                       | Sentinel-2                                             | 10   | 200x200             | [Download](http://example.com/download/3pollution)                           |                                                                       |
|                                 |                            |                                  | Sentinel-5P                                                                | 2600 | 120x120             |               |                                                                       |
| Biology                         | Wildlife detection         | Kenya                            | Airborne                                                                   | -    | 3068x4603           | [Download](http://example.com/download/kenya)                                 |                                                                       |
| Land surveying                  | LULC mapping               | C2Seg-BW                         | Gaofen-6                                                      | 10   | 256x256             | [Download](http://example.com/download/c2seg-bw)                             |                                                                       |
|                                 |                            |                                  | Gaofen-3                                                                   | 10   | 256x256             |                   |                                                                       |
|                                 | Change detection           | dsifn-cd                         | GoogleEarth                                                                | 0.3  | 512x512             | [Download](http://example.com/download/dsifn-cd)                             |                                                                       |
| Disaster management             | Flood monitoring           | Flood-3i                         | Airborne                                                                   | 0.05 | 256 × 256           | [Download](http://example.com/download/flood-3i)                             |                                                                       |
|                                 |                            | C2SMSFloods                      | Sentinel-2, Sentinel-1                                                     | 10   | 512x512             | [Download](http://example.com/download/c2smsfloods)                          |                                                                       |
|                                 | Wildfire monitoring        | CABUAR                           | Sentinel-2                                                                 | 10   | 5490 × 5490         | [Download](http://example.com/download/cabuar)                               |                                                                       |
|                                 | Landslide mapping          | GVLM                             | GoogleEarth                                                                | 0.3  | 1748x1748 ~ 10808x7424 | [Download](http://example.com/download/gvlm)                                  |                                                                       |
|                                 | Building damage assessment | xBD                              | WorldView                                                                  | 0.3  | 1024x1024           | [Download](http://example.com/download/xbd)                                   |                                                                       |

*\* for time-series data.*

## Code
We will release the code and pretrain weights after company's open-source approval.

## Acknowledgments
- [antmmf](https://github.com/alipay/Ant-Multi-Modal-Framework)
- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [Painter](https://github.com/baaivision/Painter)

