import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import json

from src.data.nmnist import get_nmnist_dataloaders
from src.models.ann_event_cnn import ANN_Event_CNN
from src.utils.device import get_device


def collapse_time(frames):
    # [B,T,C,H,W] -> [B,C,H,W]
    frames = (frames > 0).float()
    return frames.sum(dim=1)


def train_ann_nmnist(
    num_epochs=5,
    num_steps=10,
    batch_size=64,
    lr=1e-3
):
    device = get_device()

    train_loader, test_loader = get_nmnist_dataloaders(
        batch_size=batch_size,
        num_steps=num_steps
    )

    model = ANN_Event_CNN(input_channels=2, input_size=34).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    test_accuracies = []
    epoch_times = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            frames = collapse_time(frames).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(frames)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_times.append(time.time() - start_time)
        train_losses.append(total_loss)

        acc = evaluate_ann_nmnist(model, test_loader, device)
        test_accuracies.append(acc)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        print(f"Test Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "results/checkpoints/ann_nmnist_10steps.pth")

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times,
        "num_steps": num_steps
    }

    with open("results/logs/ann_nmnist_results_10steps.json", "w") as f:
        json.dump(results, f)

    return model


def evaluate_ann_nmnist(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in data_loader:
            frames = collapse_time(frames).to(device)
            labels = labels.to(device)

            outputs = model(frames)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    train_ann_nmnist()
