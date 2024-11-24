# EL-DETR

This repository is an official implementation of the paper "XXX". 


## Introduction

The code will be made fully available after the article is published online.


## Installation

### Requirements
- Python >= 3.7, CUDA >= 10.1
- PyTorch >= 1.7.0, torchvision >= 0.6.1
- Cython, COCOAPI, scipy, termcolor, numpy, matplotlib, argparse, pillow

## Usage

### Data preparation
We expect the directory structure to be the following (COCO 2017 format):
```
path/to/coco/
├── annotations/  # annotation json files
└── images/
    ├── train2017/    # train images
    ├── val2017/      # val images
    └── test2017/     # test images
```

### Training

To train EL-DETR
```shell
python main.py \
    --resume auto \
    --coco_path /path/to/coco \
    --output_dir output/conddetr_r50_epoch50
```

### Evaluation
To evaluate conditional DETR-R50 on COCO *val* with 8 GPUs run:
```shell
python main.py --eval \
    --resume <checkpoint.pth> \
    --coco_path /path/to/coco \
    --output_dir output/<output_path>
```

## License

EL-DETR is released under the Apache 2.0 license.


## Citation

```bibtex
@inproceedings{Wang2024-EL-DETR,
  title       = {EL-DETR},
  author      = {Wang et al. },
  booktitle   = {Decisoin Support Systems},
  year        = {2024}
}
```
