<h1 style="text-align: center;">🍏 PyTorch Custom ResNet for CIFAR-10 on Apple MPS 🍏</h1>

This repository implements a custom ResNet architecture using PyTorch with support for Apple MPS. The network is designed for the CIFAR-10 dataset and features a flexible architecture that lets you vary the number of layers (via residual blocks) while ensuring the total number of trainable parameters remains below 5 million.

## Custom ResNet with Variable Layers

- **Architecture:**  
  Defined in **custom_resnet.py**, the network allows you to specify the number of residual blocks per stage via a list (e.g. `[2, 3, 3, 2]`). The base number of channels (default 32) doubles at each stage.

- **Parameter Check:**  
  The helper function `count_parameters` is provided to verify that the model's parameter count is below 5 million.

## Data Handling

The data loader in **cifar_10.py** has been updated to load and preprocess the CIFAR-10 dataset (32×32 RGB images) using standard data augmentation (random cropping and horizontal flipping) and normalization.

## Training

The training script in **train.py** uses the custom ResNet model. The device is set to use Apple MPS if available, otherwise it falls back to CUDA or CPU. Before training begins, the total parameter count is printed for verification.

Example training log:
```sh
Using device: mps
Total parameters: 2,850,000
Epoch: 1 | Iter: 50 | Loss: 2.123
...
Epoch 10: Test accuracy 0.85
```

## What to do in VSCode
Open a new workspace window.

Click Clone Repository

Paste 

```
https://github.com/aaf091/ResNet_CIFAR10
```

Enter

Open the train.py file

Click on the run icon on the top right of the window

## What to do otherwise
Open terminal.
Paste the following commands
```
git clone https://github.com/aaf091/ResNet_CIFAR10.git
cd ResNet_CIFAR10
python train.py
```
