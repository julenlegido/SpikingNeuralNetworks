import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import json

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.ann_cnn import ANN_CNN
from src.utils.device import get_device


def train_ann_cifar(num_epochs=5, batch_size=64, lr=1e-3):
    device = get_device()

    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)

    model = ANN_CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []
    epoch_times = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        start_time = time.time()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        end_time = time.time()

        train_losses.append(total_loss)
        epoch_times.append(end_time - start_time)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        test_accuracy = evaluate_ann_cifar(model, test_loader, device)
        test_accuracies.append(test_accuracy)

        print(f"Test Accuracy: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), "results/checkpoints/ann_cifar.pth")

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times
    }

    with open("results/logs/ann_cifar_results.json", "w") as f:
        json.dump(results, f)

    return model


def evaluate_ann_cifar(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    train_ann_cifar()
