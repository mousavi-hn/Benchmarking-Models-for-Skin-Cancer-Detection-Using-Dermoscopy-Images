# Benchmarking-Models-for-Skin-Cancer-Detection-Using-Dermoscopy-Images

## Overview

This project presents a comprehensive benchmarking framework for Skin tumor detection using Dermo scan images. It evaluates multiple deep learning architectures — including classical convolutional neural networks (CNNs) and hybrid quantum-classical models — to analyze their performance, robustness, and scalability. I used the same data for training/validating/testing for all models, to keep it fair and compare the performance of the models only. So what I was looking for was an answer to this question: we have CNNs ready at our disposal, QNNs are a new trend, is it worth it to go for hybrid QNN-CNN ? Do they give us any advantages ?

The goal of this project is to provide a reproducible and extensible pipeline for comparing state-of-the-art models in medical image classification.

### Whole project in a glance:

* Step 1 : Trainin well known CNNs pretrained on ImageNet (VGG16/19, DenseNet121/201, etc.)
* Step 2 : Using the trained models in Step 1 for feature extraction then on top of that having quantum layers based on transfer learning technique (PennyLane + JAX)

---

## Objectives

* Benchmarking widely-used CNN architectures for tumor detection
* Explore hybrid Quantum Neural Network (QNN) + CNN models
* Evaluate models using robust metrics beyond accuracy
* Provide a reproducible and modular experimentation pipeline

---

## Models Evaluated

### Classical Models

* VGG16 / VGG19
* ResNet50V2
* DenseNet121 / DenseNet201
* EfficientNetB0
* MobileNetV2
* InceptionV3
* Xception

### Hybrid Models

* CNN feature extractor + Quantum layer (PennyLane + JAX)
* Variable number of qubits (2, 4, 6, 8, 12, 16) with depth 2

---

## Evaluation Metrics

Each model is evaluated using:

* Accuracy
* Precision
* Recall (Sensitivity)
* Specificity
* F1-score
* ROC-AUC
* Confusion Matrix

---

## Dataset

* [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
* Binary classification: **tumor / no tumor**
* I considered bcc and mel as cancer and the other types except akiec as no cancer
* I also used augmentation methods on images (cropping, rotating, etc.) to level the number of those images with cancer and those without

> ⚠️ Due to dataset licensing and privacy constraints, the data is not included in this repository.

---

## Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/mousavi-hn/Benchmarking-Models-for-Skin-Cancer-Detection-Using-Dermoscopy-Images.git
cd Benchmarking-Models-for-Skin-Cancer-Detection-Using-Dermoscopy-Images
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare dataset

* Download dataset from the provided source
* Place images in:

```
data/Dermoscopy_Images/
    ├── yes/
    ├── no/
```

### 4. Run training

```bash
python scripts/train_hybrid.py
```

---

## Project Structure

```
project/
│
├── CNN/                 
├── hybrid CNN + QNN/                    
├── results/             # Outputs (models, plots, reports)
```

### Sub-project Structure

```
sub-project/
│
├── scripts/        # Entry-point scripts
├── src/            # Core modules (data, models, training)
    ├── configs
    ├── data/
    ├── models/
    ├── train/
    ├── evaluate/

```

---

## Results

Results are automatically saved as:

* CSV summaries
* JSON reports
* Training plots (accuracy & loss curves)

Example output:

```
results/
├── saved_models/
├── plots/
├── reports/
└── hybrid_benchmark_summary.csv
```

---

## Discussion (most favourite part!)

Here I have shared the summary results, top 9 models, in a table sorted by low to high false negatives!

The important note here is this: these are vital classification models, logically false negatives are the worst to happen (patient has cancer, yet the model classifies as no cancer is present). So my endeavour was that to find those models with FNs as low as possible (ideally 0), yet as you may guess, this can not be ensured. 

| model_name     | model_type    |   n_qubits |   q_depth |   fn |   fp |   tn |   tp |   training_time_sec |   accuracy |   precision |   recall_sensitivity |   specificity |   f1_score |   roc_auc |
|:---------------|:--------------|-----------:|----------:|-----:|-----:|-----:|-----:|--------------------:|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|
| ResNet50V2     | hybrid        |         12 |         2 |    1 |    6 | 1044 |  928 |            11596.3  |   0.996463 |    0.993576 |             0.998924 |      0.994286 |   0.996243 |  0.996338 |
| VGG16          | classical CNN |        nan |       nan |    2 |    4 | 1046 |  927 |            33731.1  |   0.996968 |    0.995704 |             0.997847 |      0.99619  |   0.996774 |  0.999958 |
| DenseNet201    | classical CNN |        nan |       nan |    3 |    1 | 1049 |  926 |            21920.5  |   0.997979 |    0.998921 |             0.996771 |      0.999048 |   0.997845 |  0.999875 |
| EfficientNetB0 | hybrid        |          4 |         2 |    3 |    4 | 1046 |  926 |            14118.9  |   0.996463 |    0.995699 |             0.996771 |      0.99619  |   0.996235 |  0.999721 |
| MobileNetV2    | hybrid        |         12 |         2 |    3 |   14 | 1036 |  926 |            10818.8  |   0.99141  |    0.985106 |             0.996771 |      0.986667 |   0.990904 |  0.999552 |
| VGG19          | classical CNN |        nan |       nan |    4 |    6 | 1044 |  925 |            41627.5  |   0.994947 |    0.993555 |             0.995694 |      0.994286 |   0.994624 |  0.999836 |
| Xception       | hybrid        |          8 |         2 |    4 |    8 | 1042 |  925 |            36796.1  |   0.993936 |    0.991426 |             0.995694 |      0.992381 |   0.993555 |  0.999489 |
| InceptionV3    | hybrid        |          4 |         2 |    5 |    9 | 1041 |  924 |             6186.18 |   0.992926 |    0.990354 |             0.994618 |      0.991429 |   0.992481 |  0.992033 |
| DenseNet121    | hybrid        |          6 |         2 |    6 |    5 | 1045 |  923 |             8926.21 |   0.994442 |    0.994612 |             0.993541 |      0.995238 |   0.994076 |  0.997766 |


### Some important points about the table:
* I compared each model with its hybrid variants, I chose the one which outperforms in terms of lower false negatives
* Interestingly ResNet50V2 with only 2 qubits outperformed others significantly, the best classical model had 72 FNs (VGG19) and 739 FPs, the hybrid mentioned had 47 FNs and 747 FPs, so we gained around 35% improvement in terms of FNs and the penalty was only 1% increase in the number of FPs!
* I have provided the full tables with all the variants in TABLES.md, you find it in the main page of repository

## Key Contributions

* Unified benchmarking of multiple CNN architectures
* Integration of hybrid quantum-classical models
* Modular and reproducible ML pipeline
* Evaluation using medically relevant metrics

---

## Future Work

* Tumor segmentation (e.g., U-Net architectures)
* Model explainability (Grad-CAM, saliency maps)
* Hyperparameter optimization
* Clinical dataset validation

---

## References

* Deep Learning for Medical Image Analysis
* Quantum Machine Learning frameworks (PennyLane, JAX)
* Transfer learning in medical imaging

---

## Acknowledgments

This project is developed as part of ongoing research and study in machine learning and medical imaging.

---

## Pretrained models can be found here:
https://drive.google.com/drive/folders/1aWNdRHJqgsHlvnzahH66naxPWpQ-hzuQ?usp=sharing

* Due to high volume of the trained models, I have shared them in a google drive link! Please feel free to have a look and use the models for your work, in that case I would be glad if you please let me know about it, yet there are no licensing restrictions here, all is my independent work!

---

## Contact

For questions or collaboration:

* GitHub: https://github.com/mousavi-hn
* Email: mousavi.hn@gmail.com


