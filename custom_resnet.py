import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic residual block with batch normalization.
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Adjust shortcut if dimensions differ.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# CustomResNet with a variable number of blocks per stage.
class CustomResNet(nn.Module):
    def __init__(self, blocks_per_stage=[2, 2, 2, 2], base_channels=32, num_classes=10):
        """
        Args:
            blocks_per_stage (list): List of four integers specifying the number of residual blocks per stage.
            base_channels (int): Number of channels for the initial convolution.
            num_classes (int): Number of output classes.
        """
        super(CustomResNet, self).__init__()
        self.in_channels = base_channels
        # Initial convolution for CIFAR-10 (3-channel, 32x32 images).
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # Four stages with doubling channels.
        self.layer1 = self._make_layer(base_channels, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks_per_stage[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks_per_stage[3], stride=2)
        
        # Global average pooling and classifier.
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        # First block may downsample.
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def count_parameters(model):
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Quick test to print the total parameter count.
if __name__ == "__main__":
    model = CustomResNet(blocks_per_stage=[2, 3, 3, 2], base_channels=32, num_classes=10)
    print("Total parameters:", count_parameters(model))