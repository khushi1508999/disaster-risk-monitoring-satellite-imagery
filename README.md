# Flood Detection Using Satellite Imagery - Deep Learning with NVIDIA Triton

A complete implementation of NVIDIA's Disaster Risk Monitoring course, demonstrating automated flood detection from satellite imagery using semantic segmentation with U-Net architecture and NVIDIA Triton Inference Server deployment.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Supported-green)](https://developer.nvidia.com/tensorrt)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-DLI%20Course-76B900?logo=nvidia)](https://courses.nvidia.com/courses/course-v1:DLI+S-ES-01+V1/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

---

## Overview

Natural disasters, particularly floods, cause billions of dollars in damages and disrupt millions of lives annually. This project implements an end-to-end deep learning solution to automatically detect and segment flood events from satellite imagery, enabling faster disaster response and more effective risk monitoring.

**Key Highlights:**
- Semantic segmentation using U-Net with ResNet-18 backbone
- TensorRT optimization for high-performance inference
- Deployment-ready with NVIDIA Triton Inference Server
- Comprehensive preprocessing and evaluation pipeline
- Achieved ~73.5% precision and ~64.9% recall on test set

This implementation follows NVIDIA's Deep Learning Institute course on [Disaster Risk Monitoring Using Satellite Imagery](https://courses.nvidia.com/courses/course-v1:DLI+S-ES-01+V1/).

---

## Features

### Core Capabilities
- **Automated Flood Detection**: Pixel-wise classification of flooded vs non-flooded regions
- **Satellite Image Processing**: Handles 512x512 RGB satellite imagery
- **Real-time Inference**: Optimized TensorRT engine for production deployment
- **Performance Monitoring**: Comprehensive evaluation with IoU, precision, and recall metrics

### Technical Features
- Image preprocessing with normalization and BGR conversion
- Confusion matrix calculation for binary classification
- Model evaluation on Cloud to Street Sen1Floods11 dataset
- Production-ready inference server configuration

---

## Architecture

### U-Net Architecture
- **Backbone**: ResNet-18 (pre-trained weights)
- **Task**: Semantic Segmentation
- **Input**: 512×512×3 (RGB images)
- **Output**: 512×512×1 (Binary flood mask)
- **Classes**: 2 (flood, no-flood)

---

## 📊 Dataset

**Source**: [Cloud to Street Sen1Floods11 Dataset](https://github.com/cloudtostreet/Sen1Floods11)

- **Type**: Synthetic Aperture Radar (SAR) and optical satellite imagery
- **Coverage**: Global flood events from Sentinel-1 mission
- **Format**: PNG images and corresponding binary masks
- **Split**: Training, validation, and test sets
- **Resolution**: 512×512 pixels


---

## 🚀 Installation

### Prerequisites
- **Operating System**: Linux (Ubuntu 18.04+)
- **GPU**: NVIDIA GPU with CUDA support
- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher
- **Docker**: For Triton Inference Server (optional but recommended)


### Step 2: Install Dependencies
## Install Python packages

```
pip install -r requirements.txt
```

## Install NVIDIA Triton Client
```
pip install tritonclient[all]
```


---

## 📈 Model Performance

### Metrics on Test Set

| Metric | Value |
|--------|-------|
| **Precision** | 73.54% |
| **Recall** | 64.86% |
| **IoU (Intersection over Union)** | 52-55% |

### Confusion Matrix


**Formulas:**
- **Precision** = TP / (TP + FP) - How many flood predictions were correct
- **Recall** = TP / (TP + FN) - How many actual floods were detected
- **IoU** = TP / (TP + FP + FN) - Overall segmentation accuracy

---

## Technologies Used

### Deep Learning Frameworks
- **TensorFlow** / **PyTorch** - Model training framework
- **NVIDIA TAO Toolkit** - Transfer learning and model optimization
- **TensorRT** - Model optimization for inference

### Deployment
- **NVIDIA Triton Inference Server** - Production model serving
- **Docker** - Containerized deployment

### Data Processing
- **NVIDIA DALI** - GPU-accelerated data loading and preprocessing
- **NumPy** - Numerical computing
- **Pillow (PIL)** - Image processing
- **OpenCV** - Computer vision utilities

### Visualization & Analysis
- **Matplotlib** - Plotting and visualization
- **Pandas** - Data analysis
- **Jupyter Notebook** - Interactive development

---

## Results

### Key Achievements
- Successfully trained U-Net segmentation model on satellite flood imagery  
- Deployed optimized TensorRT model on Triton Inference Server  
- Achieved production-ready inference performance  
- Implemented complete evaluation pipeline with confusion matrix metrics  
- Created reproducible workflow for disaster risk monitoring  

### Insights
- The model performs well on clear flood boundaries
- Challenges remain in distinguishing shadows from water bodies
- Model generalization could be improved with more diverse training data
- TensorRT optimization provides significant inference speedup

---
## 🙏 Acknowledgments

- **NVIDIA Deep Learning Institute** for the comprehensive course material and training resources
- **Cloud to Street** for providing the [Sen1Floods11 dataset](https://github.com/cloudtostreet/Sen1Floods11)
- **ESA Copernicus Program** for Sentinel-1 satellite data
- Course instructors and the NVIDIA AI research team

### References
- [NVIDIA Deep Learning Institute - Disaster Risk Monitoring](https://courses.nvidia.com/courses/course-v1:DLI+S-ES-01+V1/)
- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Note**: While this implementation is open-source, the original NVIDIA DLI course materials are property of NVIDIA Corporation. This repository contains my personal implementation and notes from completing the course.

---

