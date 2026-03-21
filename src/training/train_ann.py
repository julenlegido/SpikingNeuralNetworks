import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.mnist import get_mnist_dataloaders
from src.models.ann_mlp import ANN_MLP
from src.utils.device import get_device

from carbontracker.tracker import CarbonTracker # Energy usage, CO2 estimation, power consumption


def train_ann(num_epochs=10, num_steps=50, batch_size=64, lr=1e-3):
    device = get_device()

    # Load data
    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)

    # Model
    model = ANN_MLP().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tracker = CarbonTracker(epochs=num_epochs) # CarbonTracker

    for epoch in range(num_epochs):
        tracker.epoch_start()
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass (no time loop)
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Evaluate
        test_accuracy = evaluate_ann(model, test_loader, device)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        tracker.epoch_end()

    tracker.stop()
    return model


def evaluate_ann(model, data_loader, device):
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
    train_ann()