# Semantic-Guided Monocular Depth Estimation Based on Patch Knowledge Distillation

## Introduction
This repository contains the implementation of the Semantic-Guided Monocular Depth Estimation model based on Patch Knowledge Distillation, developed by Wang Bin from Nanjing University of Information Science and Technology.

The model enhances depth estimation accuracy through explicit constraints from semantic segmentation, focusing on refining depth estimation contours.

### Result3

| Method | Train | BackBone | Abs Rel | Sq Rel | RMSE | RMSE log |δ < 1.25 | δ² < 1.25 | δ³ < 1.25 | 
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| SGDepth | M+Se | ResNet50 | 0.112 | 0.785 | 4.537 | 0.189 | 0.885 | 0.963 |0.982   |
| Ours | M+Se |ResNet50|0.103|0.685|4.565|0.177|0.887|0.963|0.985|
| Ours(288x960) | M+Se|ResNet50|0.099|0.671|4.522|0.174|0.891|0.964|0.985 |
| Ours(384x1280) |M+Se|ResNet50|0.094|0.645|4.451|0.171|0.894|0.966|0.986|

## Modules

### 1. Semantic-Depth Consistency Loss
We devised a semantic-guided monocular depth estimation model based on patch knowledge distillation (SGDPKD). This model incorporates a semantic depth consistency loss, ensuring that pixels with the same semantic label possess consistent relative depth.

### 2. Patch Knowledge Distillation Module
We improved the patch-based depth refinement method by employing dual networks (self+DPT\citep{yifan2019patch}) to estimate multi-scale patch depths. These patch depths are fused using contour masks. We incorporate this approach into model training, achieving patch-based knowledge distillation via the patch loss function.

### 3. Block Self-Attention Mechanism
We introduced a block-wise self-attention module in the decoder to enhance the extraction capability of deep features. Compared to conventional attention modules, the block-wise approach reduces the computational load of the model.

## Usage
1. **Installation**: Follow the installation instructions in `INSTALL.md`.
2. **Training**: Run `train.py` with specified parameters to train the model.
3. **Evaluation**: Evaluate model performance using `evaluate.py` on test datasets.
4. **Inference**: Perform depth estimation on new images with `inference.py`.

## Requirements
- Python 3.9
- Dependencies listed in `requirements.txt`

## Citation
If you find this work helpful in your research, please cite:
```bibtex
@article{wang2024semanticdepth,
  title={Semantic-Guided Monocular Depth Estimation Based on Patch Knowledge Distillation},
  author={Wang, Bin},
  journal={Journal of Computer Vision},
  year={2024}
}

