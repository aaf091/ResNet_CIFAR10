import torch
import torch.optim as optim
import torch.nn as nn
from custom_resnet import CustomResNet, count_parameters
from cifar_10 import get_data  # This file now loads CIFAR-10

def train(model, train_loader, optimizer, criterion, device, epoch, print_interval=50):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % print_interval == 0:
            print(f"Epoch: {epoch} | Iter: {i} | Loss: {running_loss / print_interval:.3f}")
            running_loss = 0.0

def test(model, test_loader, device, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(f"Epoch {epoch}: Test accuracy {accuracy:.3f}")
    return accuracy

if __name__ == "__main__":
    # Set device to Apple MPS if available; otherwise, use CUDA or CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    train_loader, test_loader = get_data(batch_size=128)
    
    # Experiment with different configurations; here we use [2, 3, 3, 2] as an example.
    model = CustomResNet(blocks_per_stage=[2, 3, 3, 2], base_channels=32, num_classes=10).to(device)
    
    total_params = count_parameters(model)
    print("Total parameters:", total_params)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 50  # Adjust epochs as needed.
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        test(model, test_loader, device, epoch)