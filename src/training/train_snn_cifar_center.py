import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import json

from snntorch import spikegen

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.snn_cnn import SNN_CNN
from src.utils.device import get_device
from src.utils.center_weight_mask import create_center_weight_mask


def train_snn_cifar_center(num_epochs=5, num_steps=10, batch_size=64, lr=1e-3):
    device = get_device()

    train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)

    model = SNN_CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # CREATE MASK FOR CIFAR (32x32)
    mask = create_center_weight_mask(size=32).to(device)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,32,32]

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

            # APPLY CENTER WEIGHTING (broadcasts to 3 channels)
            weighted_images = images * mask

            spike_input = spikegen.rate(weighted_images, num_steps=num_steps)

            optimizer.zero_grad()

            spk_rec = model(spike_input)
            spk_sum = spk_rec.sum(dim=0)

            loss = criterion(spk_sum, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        end_time = time.time()

        train_losses.append(total_loss)
        epoch_times.append(end_time - start_time)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        test_accuracy = evaluate_snn_cifar_center(
            model, test_loader, device, num_steps, mask
        )

        test_accuracies.append(test_accuracy)

        print(f"Test Accuracy: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), "results/checkpoints/snn_cifar_center.pth")

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times,
        "num_steps": num_steps
    }

    with open("results/logs/snn_cifar_center_results.json", "w") as f:
        json.dump(results, f)

    return model


def evaluate_snn_cifar_center(model, data_loader, device, num_steps, mask):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            weighted_images = images * mask
            spike_input = spikegen.rate(weighted_images, num_steps=num_steps)

            spk_rec = model(spike_input)
            spk_sum = spk_rec.sum(dim=0)

            _, predicted = spk_sum.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    train_snn_cifar_center()