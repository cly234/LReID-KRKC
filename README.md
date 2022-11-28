# [AAAI2023] Lifelong Person Re-Identification via Knowledge Refreshing and Consolidation
The official implementation of AAAI 2023 paper "Lifelong Person Re-Identification via Knowledge Refreshing and Consolidation" by Chunlin Yu, Ye Shi, Zimo Liu, Shenghua Gao, Jingya Wang*
## Method Overview
![](./docs/KRKC_fig.png)
## Getting Started
### Data preparation
- Prepare the dataset structure as follows
- Move docs/splits.json file into directory /path/to/your/dataset/viper
- Covert CUHK-SYSU to cuhksysu4reid following instructions in this [repo](https://github.com/TPCD/LifelongReID)
```
/path/to/your/dataset
├── market1501
│   │── bounding_box_test
│   │── bounding_box_train
│   └── ...
├── cuhksysu4reid
│   │── combine
│   │── gallery
│   └── ...
├── MSMT17
│   │── test
│   │── train
│   └── ...
├── viper
│   │── VIPeR
│   │   │── cam_a
│   │   └── cam_b 
│   └── splits.json
```  	
## Run the Code
CUDA_VISBILE_DEVICES=0,1,2,3 python continual_train.py --data-dir=$ROOT

## Acknowledgement
- The code framework is based on [PTKP](https://github.com/g3956/PTKP).
- The code for evaluation is borrowed from [TransReID](https://github.com/damo-cv/TransReID).
