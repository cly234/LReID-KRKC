# [AAAI2023] Lifelong Person Re-Identification via Knowledge Refreshing and Consolidation
The official implementation of AAAI 2023 paper "[Lifelong Person Re-Identification via Knowledge Refreshing and Consolidation](https://arxiv.org/abs/2211.16201)" by Chunlin Yu, Ye Shi, Zimo Liu, Shenghua Gao, Jingya Wang*

## Introduction
Lifelong person re-identification (LReID) is in significant demand for real-world development as a large amount of ReID data is captured from diverse locations over time and cannot be accessed at once inherently. However, a key challenge for LReID is how to incrementally preserve old knowledge and gradually add new capabilities to the system. Unlike most existing LReID methods, which mainly focus on dealing with catastrophic forgetting, our focus is on a more challenging problem, which is, not only trying to reduce the forgetting on old tasks but also aiming to improve the model performance on both new and old tasks during the lifelong learning process. Inspired by the biological process of human cognition where the somatosensory neocortex and the hippocampus work together in memory consolidation, we formulated a model called Knowledge Refreshing and Consolidation (KRC) that achieves both positive forward and backward transfer. More specifically, a knowledge refreshing scheme is incorporated with the knowledge rehearsal mechanism to enable bi-directional knowledge transfer by introducing a dynamic memory model and an adaptive working model. Moreover, a knowledge consolidation scheme operating on the dual space further improves model stability over the long term. Extensive evaluations show KRC’s superiority over the state-of-the-art LReID methods on challenging pedestrian benchmarks.
![](./docs/KRKC_fig.png)
## Getting Started
### Requirements
- Python 3.6+
- Pytorch 1.9.0
- For more detailed requirements, run
```
pip install -r requirements.txt
```
### Dataset preparation
- Prepare the dataset structure as in [here](https://github.com/cly234/LReID-KRKC/blob/main/docs/dataset_structure.md).
- Move docs/splits.json file into directory /path/to/your/dataset/viper.
- Convert CUHK-SYSU to cuhksysu4reid following instructions in this [repo](https://github.com/TPCD/LifelongReID).
### Training
```
CUDA_VISBILE_DEVICES=0,1,2,3 python continual_train.py --data-dir=/path/to/your/dataset
```

### Evaluation
```
python evaluate.py --data-dir=/path/to/your/dataset --resume-working=/path/to/working/checkpoints --resume-memory=/path/to/memory/checkpoints
```
### Checkpoints
We provide the checkpoints of working model and memory model trained after the last step in [Google Drive](https://drive.google.com/drive/folders/1rDFTr7jsLrxnMMFL54meB03CkXu5Yh63?usp=sharing).
## Acknowledgement
Thanks for all these great code bases:
- The code framework is based on [PTKP](https://github.com/g3956/PTKP) and [AKA](https://github.com/TPCD/LifelongReID).
- The code for efficient evaluation is borrowed from [TransReID](https://github.com/damo-cv/TransReID).
## Cite this work
```
@article{yu2022lifelong,
  title={Lifelong Person Re-Identification via Knowledge Refreshing and Consolidation},
  author={Chunlin Yu and Ye Shi and Zimo Liu and Shenghua Gao and Jingya Wang},
  journal={arXiv preprint arXiv:2211.16201},
  year={2022}
}
```
