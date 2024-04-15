# Applying Camouflaged Object Detection Modules to Microscopic Histopathology Images for Weakly Supervised Semantic Segmentation

### Weakly Supervised Learning for Microscopic Histopathology Image Segmentation

## 1. Abstract
Deep learning-based image segmentation is one of the significant tasks in the field of histopathology analysis. Since pixel-wise label generation is essential for training segmentation networks, segmentation learning requires time and resources. Therefore, Weakly Supervised Learning (WSL) methods using only image labels, which are relatively less burdensome to generate labels, have been actively studied.In particular, the importance of WSL is increasing in the medical field, where label annotation costs are high. However, pathological images with ambiguous boundaries between tumors and backgrounds make it difficult to apply existing WSL methods targeting natural images effectively. We propose a new approach to effectively segment tumors from pathology images only using image labels by combining Camouflaged Object Detection networks (COD networks) feature generation modules to an FCN for classification. We apply the Receive Field Block, Feature Aggregation, and Attention module, the components that make up the COD, to the FCN for advanced Class Activation Map (CAM). In addition, our model remarkably outperformed the previous WSL methods in microscopic histopathology.

Keywords : Histopathology Analysis, Microscopic Image, Weakly Supervised Learning, Segmentation, Camouflaged Object Detection   
     

## 2. Method
<img src = "https://github.com/TaeMiKim/WSL-for-Colon-Histopathology/blob/main/figures/Architecture.png" width="800" height="400">

<p align="center">
<img src = "https://github.com/TaeMiKim/WSL-for-Colon-Histopathology/blob/main/figures/RF.png" width="300" height="300">
</p>

<p align="center">
<img src = "https://github.com/TaeMiKim/WSL-for-Colon-Histopathology/blob/main/figures/AGG.png" width="300" height="300">
</p>

<p align="center">
<img src = "https://github.com/TaeMiKim/WSL-for-Colon-Histopathology/blob/main/figures/HA.png" width="600" height="200">
</p>


## 3. Results
<p align="center">
<img src = "https://github.com/TaeMiKim/WSL-for-Colon-Histopathology/blob/main/figures/Result.png" width="700" height="300">
</p>

<p align="center">
<img src = "https://github.com/TaeMiKim/WSL-for-Colon-Histopathology/blob/main/figures/result_table.png" width="500" height="200">
</p>

## 5. Contact
TaeMi Kim, xoal9797@gmail.com
