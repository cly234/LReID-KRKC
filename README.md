# LReID-KRKC
The code implementation of "Lifelong Person Re-Identification via Knowledge Refreshing and Knowledge Consolidation"
## Prepare Datasets
$ROOT/LReID-KRKC  
$ROOT/data  
├── market1501  
&emsp &emsp├── bounding_box_test  
&emsp &emsp...  
└── cuhksysu4reid  
&emsp&emsp├── train  
         ...  
└── MSMT17  
 >>├── train  
         ...  
└── viper  
 >>├── VIPeR    
>>>>├── cam_a  
>>>>├── cam_b  
         ...  	
## Run the Code
CUDA_VISBILE_DEVICES=0,1,2,3 python continual_train.py
