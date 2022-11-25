# [AAAI2023] Lifelong Person Re-Identification via Knowledge Refreshing and Consolidation
The official implementation of AAAI 2023 paper "Lifelong Person Re-Identification via Knowledge Refreshing and Consolidation" by Chunlin Yu, Ye Shi, Zimo Liu, Shenghua Gao, Jingya Wang*
## Method Overview
## Prepare Datasets
```
$ROOT
├──reid
├──data
│  ├── market1501
│  │   │── bounding_box_test
│  │   │── bounding_box_train
│  │   └── ...
│  ├── cuhksysu4reid
│  │   │── combine
│  │   │── gallery
│  │   └── ...
│  ├── MSMT17
│  │   │── test
│  │   │── train
│  │   └── ...
│  ├── viper
│  │   │── VIPeR
│  │   │   │── cam_a
│  │   │   └── cam_b 
│  │   └── splits.json
```  	
## Run the Code
CUDA_VISBILE_DEVICES=0,1,2,3 python continual_train.py --data-dir=$ROOT

## Acknowledgement
- The code framework is based on [PTKP](https://github.com/g3956/PTKP).
- The code for evaluation is borrowed from [TransReID](https://github.com/damo-cv/TransReID).
