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

Here I have shared the summary results, top 9 models, in a table sorted by low to high false negatives.

| model_name     | model_type    |   n_qubits |   q_depth |   fn |   fp |   tn |   tp |   training_time_sec |   accuracy |   precision |   recall_sensitivity |   specificity |   f1_score |   roc_auc |
|:---------------|:--------------|-----------:|----------:|-----:|-----:|-----:|-----:|--------------------:|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|
| ResNet50V2     | hybrid        |          2 |         2 |   47 |  747 |  463 |  750 |            10275.4  |   0.604385 |    0.501002 |             0.941029 |      0.382645 |   0.65388  |  0.786641 |
| VGG19          | hybrid        |          2 |         2 |   52 |  874 |  336 |  745 |            29198    |   0.538615 |    0.460161 |             0.934755 |      0.277686 |   0.616722 |  0.779889 |
| VGG19          | hybrid        |          8 |         2 |   71 |  771 |  439 |  726 |            41108.3  |   0.580468 |    0.48497  |             0.910916 |      0.36281  |   0.632956 |  0.751474 |
| VGG19          | classical CNN |        nan |       nan |   72 |  739 |  471 |  725 |            38063    |   0.595914 |    0.495219 |             0.909661 |      0.389256 |   0.641309 |  0.784895 |
| VGG16          | hybrid        |         12 |         2 |   86 |  650 |  560 |  711 |            25261.3  |   0.633284 |    0.52241  |             0.892095 |      0.46281  |   0.658943 |  0.773017 |
| ResNet50V2     | hybrid        |         16 |         2 |   89 |  589 |  621 |  708 |            17084.9  |   0.662182 |    0.545875 |             0.888331 |      0.513223 |   0.676218 |  0.777859 |
| MobileNetV2    | hybrid        |         12 |         2 |    3 |   14 | 1036 |  926 |            10818.8  |   0.99141  |    0.985106 |             0.996771 |      0.986667 |   0.990904 |  0.999552 |
| VGG16          | hybrid        |          2 |         2 |   91 |  691 |  519 |  706 |            32132.1  |   0.610364 |    0.505369 |             0.885822 |      0.428926 |   0.643573 |  0.724729 |           
| VGG16          | hybrid        |          6 |         2 |   93 |  708 |  502 |  704 |            29428.8  |   0.600897 |    0.498584 |             0.883312 |      0.414876 |   0.637392 |  0.762377 |           
| VGG19          | hybrid        |          4 |         2 |   94 |  611 |  599 |  703 |            41084.6  |   0.648729 |    0.535008 |             0.882058 |      0.495041 |   0.666035 |  0.7971   |
| VGG19          | hybrid        |          6 |         2 |   95 |  613 |  597 |  702 |            41083.6  |   0.647235 |    0.53384  |             0.880803 |      0.493388 |   0.664773 |  0.80092  |


### Some important points about the table:
* In top 9 models we have 8 hybrids and just 1 classical CNN
* ResNet50V2 with 2 qubits and depth 2 shows a roughly 35% decrease in number of false negatives vs VGG19 which had the best performance among classical CNNs
* The above two points shows clearly that performance increases significantly by adding a quantum layer to our classical CNNs
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


