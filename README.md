# LReID-KRKC
The code implementation of "Lifelong Person Re-Identification via Knowledge Refreshing and Knowledge Consolidation"
## Prepare Datasets
```
data
├── market1501
│   │── bounding_box_test
│   │   └── images
│   └── *.txt
├── cuhksysu4reid
│   │── class_name
│   │   └── images
│   └── *.txt
├── MSMT17
│   │── class_name
│   │   └── images
│   └── *.txt
├── viper
│   │── VIPeR
│   │   │── class_name
│   │   │   └── images
│   │   └── *.txt 
│   └── validation
│       │── class_name
│       │   └── images
│       └── *.txt 
```
$ROOT/LReID-KRKC  
$ROOT/data  
├── market1501  
&emsp; &emsp;├── bounding_box_test  
&emsp; &emsp; ...  
└── cuhksysu4reid  
&emsp;&emsp; ├── train  
 &emsp; &emsp; ...  
└── MSMT17  
 &emsp; &emsp; ├── train  
 &emsp; &emsp; ...  
└── viper  
 &emsp; &emsp; ├── VIPeR    
&emsp; &emsp; &emsp; &emsp; ├── cam_a  
&emsp; &emsp; &emsp; &emsp; ├── cam_b  
 &emsp; &emsp; ...  	
## Run the Code
CUDA_VISBILE_DEVICES=0,1,2,3 python continual_train.py
