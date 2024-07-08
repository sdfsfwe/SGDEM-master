# Semantic-Guided Monocular Depth Estimation Based on Patch Knowledge Distillation

## Introduction
This repository contains the implementation of the Semantic-Guided Monocular Depth Estimation model based on Patch Knowledge Distillation, developed by Wang Bin from Nanjing University of Information Science and Technology.

The model enhances depth estimation accuracy through explicit constraints from semantic segmentation, focusing on refining depth estimation contours.

### Results

| Method | Train | BackBone | Abs Rel | Sq Rel | RMSE | RMSE log |  | Value 8 | Value 9 | Value 10 |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|
| SGDepth | M+Se | ResNet50 | 0.112 |  | 0.82 | 0.88 | 0.85 | 0.82 | 0.88 | 0.85 |
| Ours | 0.85 | 0.82 | 0.88 | 0.85 | 0.82 | 0.88 | 0.85 | 0.82 | 0.88 | 0.85 |
| Metric 3 | 0.85 | 0.82 | 0.88 | 0.85 | 0.82 | 0.88 | 0.85 | 0.82 | 0.88 | 0.85 |
| Metric 4 | 0.85 | 0.82 | 0.88 | 0.85 | 0.82 | 0.88 | 0.85 | 0.82 | 0.88 | 0.85 |

## Modules

### 1. Semantic-Depth Consistency Loss
Description: Enhances depth estimation by leveraging semantic segmentation information to ensure consistency between depth predictions and semantic segmentation labels.

### 2. Patch Knowledge Distillation Module
Description: Transfers knowledge from high-resolution patches to refine depth predictions, improving fine details in depth maps.

### 3. Block Self-Attention Mechanism
Description: Integrates a self-attention mechanism at the block level to capture long-range dependencies and improve global coherence in depth estimation.

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

