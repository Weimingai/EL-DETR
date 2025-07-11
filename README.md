# EL-DETR

This repository is an official implementation of the paper "An explainable lesion detection transformer model for medical imaging diagnosis decisionsupport: Design science research". (has been accepted to Decision Support Systems)


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
    --coco_path /your_coco_dataset_path \
    --output_dir output/
```

### Evaluation
To evaluate El-DETR
```shell
python main.py --eval \
    --resume <checkpoint.pth> \
    --coco_path /your_coco_dataset_path \
    --output_dir output/<output_path>
```

## License

EL-DETR is released under the Apache 2.0 license. - see the [LICENSE](./LICENSE) file for details.


## Citation

```bibtex
@inproceedings{ Wang2025-EL-DETR,
  title       = {An explainable lesion detection transformer model for medical imaging diagnosis decisionsupport: Design science research},
  author      = {Wang et al. },
  booktitle   = {Decisoin Support Systems},
  year        = {2025},
  doi         = {https://doi.org/10.1016/j.dss.2025.114492}
}
```
