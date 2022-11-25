# Lifelong Person Re-Identification via Knowledge Refreshing and Consolidation
Implementation for AAAI2023 paper "Lifelong Person Re-Identification via Knowledge Refreshing and Consolidation"
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

