<h1 style="text-align: center;">PyTorch Custom ResNet for CIFAR-10</h1>
This repository contains a submission for NYU ECE-GY 7123 Deep Learning S25 Kaggle Competition authored by Aadit Fadia, Riya Shah and Isha Math.

## Overview
We have proposed a Custom ResNet bottleneck model for CIFAR-10 classification, achieving 94.60% test accuracy under constrained parameters.

## Model Specifications
- **Model**: Custom ResNet Bottleneck
- **Number of Parameters**: 4,995,338
- **Dataset**: CIFAR-10
- **Epochs**: 180
- **Batch Size**: 128
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduler**: ReduceLROnPlateau (Factor: 0.5, Patience: 5 epochs)
- **Final Test Accuracy**: 94.60%

## Usage
1. Run the `Final_Resnet_CIFAR.ipynb` notebook

## Authors
- Aadit Fadia
- Riya Shah
- Isha math


