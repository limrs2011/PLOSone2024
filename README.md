# Adaptive Fusion U-Nets for Brain Tissue Segmentation in Non-Contrast Brain CT

This repository provides the source code associated with our PLOS ONE article:

**Refining CT image analysis: Exploring adaptive fusion in U-nets for enhanced brain tissue segmentation**
Bang-Chuan Chen, Chung-Yi Shen, Jyh-Wen Chai, Ren-Hung Hwang, Wei-Chuan Chiang, Chi-Hsiang Chou, and Wei-Min Liu
*PLOS ONE*, 20(6): e0323692, 2025
DOI: https://doi.org/10.1371/journal.pone.0323692

## Overview

Non-contrast computed tomography (NCCT) is widely used in emergency stroke assessment because it is fast and clinically accessible. However, deep-learning-based infarction lesion segmentation may produce false alarms outside the cerebral region. To reduce these false alarms, this project focuses on **brain tissue segmentation (BTS)** and proposes an **adaptive result-fusion strategy** for U-Net-family segmentation models.

The core idea is to combine segmentation outputs from multiple U-Net variants so that the final brain mask better confines downstream lesion analysis to cerebral tissue. In our study, the fusion of **UNet++ (denoted as UNet2+)** and **UNet3+** achieved the best overall performance on NCCT brain tissue segmentation.

## Key Features

* Brain tissue segmentation for non-contrast brain CT images
* Implementations of U-Net-family segmentation models:

  * U-Net
  * UNet++ / UNet2+
  * UNet3+
* Adaptive result-fusion strategy for reducing false alarms
* Optional Gaussian-filter-based post-processing
* Evaluation using:

  * Intersection over Union (IoU)
  * Hausdorff Distance (HD)
* Experimental support for testing fusion behavior on non-medical segmentation data

## Method Summary

The proposed workflow consists of the following steps:

1. **Input preparation**
   NCCT slices and corresponding binary brain tissue masks are prepared for training and evaluation.

2. **Model training**
   U-Net, UNet2+, and UNet3+ are trained separately using binary cross-entropy loss. Data augmentation may include random rotation, vertical flipping, and horizontal flipping.

3. **Segmentation prediction**
   Each trained model generates a binary brain tissue mask.

4. **Adaptive result fusion**
   The predicted masks are fused according to the segmentation behavior of the models. If models tend to produce more over-segmented pixels than missed brain pixels, an intersection operation is used to suppress false positives.

5. **Post-processing**
   A 9×9 Gaussian filter with unit standard deviation can be applied, followed by binarization using a threshold of 0.5.

6. **Evaluation**
   The final masks are evaluated using IoU and HD.

## Main Results

On the NCCT brain tissue segmentation task, the best-performing fusion strategy was obtained by combining **UNet2+ and UNet3+**.

| Method                                 |    IoU |   HD |
| -------------------------------------- | -----: | ---: |
| U-Net                                  |  0.937 | 4.52 |
| UNet2+                                 |  0.903 | 24.9 |
| UNet3+                                 |  0.948 | 1.69 |
| Fusion: UNet2+ + UNet3+                | 0.9550 | 1.33 |
| Fusion: UNet2+ + UNet3+ with filtering | 0.9552 | 1.55 |

The fusion strategy reduced excessive non-brain segmentation areas and improved overall segmentation accuracy compared with individual U-Net-family models.

## Environment

The original experiments were conducted using the following environment:

* OS: Ubuntu 18.04.1 LTS, 64-bit
* CPU: 2 × Intel Xeon Silver 4110
* GPU: 2 × NVIDIA RTX 2080 Ti, 11 GB
* Memory: 314 GB
* CUDA: 10.2
* cuDNN: 7.6.5
* Python: 3.7.9
* PyTorch: 1.7.0
* OpenCV-Python: 4.1.2

A similar GPU-enabled PyTorch environment should also be usable, although exact reproducibility may depend on CUDA, cuDNN, PyTorch, and hardware versions.


## Data Availability

The clinical brain CT data used in the study are not publicly available due to ethical restrictions concerning patient confidentiality and privacy. Researchers interested in data access should follow the procedure described in the published article and contact the relevant Institutional Review Board for eligibility review.

## Medical Disclaimer

This repository is intended for research purposes only. It is not a certified medical device and should not be used for clinical diagnosis, treatment planning, or patient management without appropriate validation, regulatory approval, and clinical supervision.

## License

Please see the `LICENSE` file for the license terms of this repository.

The associated article is open access under the Creative Commons Attribution License.
