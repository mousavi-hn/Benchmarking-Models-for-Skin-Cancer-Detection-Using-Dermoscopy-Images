# Benchmarking-Models-for-Skin-Cancer-Detection-Using-Dermoscopy-Images

## Overview

This project presents a comprehensive benchmarking framework for skin-lesion classification using dermoscopy images. It evaluates multiple deep learning architectures — including classical convolutional neural networks (CNNs) and hybrid quantum-classical models — to analyze their performance, robustness, and scalability. I used consistent dataset partitions for comparing the models, to keep it fair and compare the performance of the models only. So what I was looking for was an answer to this question: we have CNNs ready at our disposal, QNNs are a new trend, is it worth it to go for hybrid QNN-CNN ? Do they give us any advantages ?

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
* Binary classification: **melanoma or basal-cell carcinoma / selected other lesion categories**
* The positive class includes **bcc** and **mel**; the negative class excludes **akiec** and follows the selected nonmalignant lesion categories in the prepared dataset
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
python Hybrid/scripts/train_hybrid.py
```

---

## Project Structure

```
project/
│
├── CNN/                 
├── Hybrid/                    
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
├── Dermo_cnn_benchmark_results/
└── Dermo_hybrid_benchmark_results/
    ├── plots/
    ├── reports/
    └── hybrid_benchmark_summary.csv
```

---

## Discussion (most favourite part!)

### Top 9 configurations ranked by false negatives

The ranking below combines **all 9 classical CNNs and all 54 hybrid configurations** reported in [TABLES.md](TABLES.md). Configurations are ordered by **ascending false negatives (FN)** on the reported test set, with **ascending false positives (FP)** as a tie-breaker. This is an FN-prioritized ranking, **not** a ranking by overall accuracy, balanced accuracy, or clinical utility.

| Rank | Model | Type | Qubits | Depth | FN | FP | TP | TN | Sensitivity | Specificity | F1 | Balanced acc. | ROC-AUC |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ResNet50V2 | Hybrid | 2 | 2 | 47 | 747 | 750 | 463 | 94.10% | 38.26% | 65.39% | 66.18% | 78.66% |
| 2 | VGG19 | Hybrid | 2 | 2 | 52 | 874 | 745 | 336 | 93.48% | 27.77% | 61.67% | 60.62% | 77.99% |
| 3 | VGG19 | Hybrid | 8 | 2 | 71 | 771 | 726 | 439 | 91.09% | 36.28% | 63.30% | 63.69% | 75.15% |
| 4 | VGG19 | Classical | — | — | 72 | 739 | 725 | 471 | 90.97% | 38.93% | 64.13% | 64.95% | 78.49% |
| 5 | VGG16 | Hybrid | 12 | 2 | 86 | 650 | 711 | 560 | 89.21% | 46.28% | 65.89% | 67.75% | 77.30% |
| 6 | ResNet50V2 | Hybrid | 16 | 2 | 89 | 589 | 708 | 621 | 88.83% | 51.32% | 67.62% | 70.08% | 77.79% |
| 7 | VGG16 | Hybrid | 2 | 2 | 91 | 691 | 706 | 519 | 88.58% | 42.89% | 64.36% | 65.74% | 72.47% |
| 8 | VGG16 | Hybrid | 6 | 2 | 93 | 708 | 704 | 502 | 88.33% | 41.49% | 63.74% | 64.91% | 76.24% |
| 9 | VGG19 | Hybrid | 4 | 2 | 94 | 611 | 703 | 599 | 88.21% | 49.50% | 66.60% | 68.85% | 79.71% |

### Key observations

- **Lowest FN, classical:** VGG19, with **72 FN** and **739 FP**.
- **Lowest FN, hybrid:** ResNet50V2 with **2 qubits**, depth **2**, with **47 FN** and **747 FP**. This is **25 fewer FN (34.72% relative reduction)** than the best classical configuration, but **8 more FP**.
- **Top nine:** **8 hybrid configurations and 1 classical CNN**, under the explicitly stated FN-based ranking. Multiple configurations of the same backbone count separately.
- **Trade-off:** The selected hybrid's sensitivity increases from **90.97%** to **94.10%**, while specificity decreases from **38.93%** to **38.26%**. F1 improves from **64.13%** to **65.39%** and balanced accuracy from **64.95%** to **66.18%**. The false-positive burden remains substantial.
- **Interpretation:** These are descriptive results for the reported dataset and selection procedure; they do **not** establish that quantum layers universally improve CNNs or demonstrate clinical readiness. The complete configuration-level results are available in [TABLES.md](TABLES.md).

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


