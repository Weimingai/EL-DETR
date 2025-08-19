# EL-DETR

This repository is an official implementation of the paper "An explainable lesion detection transformer model for medical imaging diagnosis decisionsupport: Design science research". (has been accepted to Decision Support Systems)


## Introduction

Utilizing machine learning methods for auxiliary decision support in medical imaging significantly reduces missed detections and unnecessary expenses. However, the strict accuracy and transparency requirements in the medical field pose challenges for deep learning applications based on neural networks. To address these issues, we propose a novel artificial intelligence artifact guided by the design science research methodology for lesion detection decision support in medical images, called Explainable Lesion DEtection TRansformer (EL-DETR). This approach features an explainable separate attention mechanism in the decoder that highlights the attention weights of content and location queries, providing insights into the inference process through attention mapping visualizations. In addition, we introduce a hybrid matching query strategy to enhance the learning of positive samples and develop an adaptive efficient compound loss function to optimize training. We demonstrate ELDETR's superior accuracy, robustness, and interpretability using four real-world datasets, establishing it as a reliable tool for clinical diagnosis and treatment decision support based on medical imaging.


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
