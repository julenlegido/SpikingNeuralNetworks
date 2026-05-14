import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import time
import json

from src.data.nmnist import get_nmnist_dataloaders
from src.models.snn_cnn_nmnist import SNN_CNN
from src.utils.device import get_device


def train_snn_nmnist(
    num_epochs=5,
    num_steps=15,
    batch_size=64,
    lr=1e-3,
    beta=0.9
):

    device = get_device()

    train_loader, test_loader = get_nmnist_dataloaders(
        batch_size=batch_size,
        num_steps=num_steps
    )

    model = SNN_CNN(beta=beta).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    train_losses = []
    test_accuracies = []
    epoch_times = []

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0

        start_time = time.time()

        for frames, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        ):

            frames = frames.float().to(device)
            labels = labels.to(device)

            # [B,T,C,H,W] -> [T,B,C,H,W]
            frames = frames.permute(1,0,2,3,4)

            optimizer.zero_grad()

            spk_rec = model(frames)

            spk_sum = spk_rec.sum(dim=0)

            targets = torch.zeros(
                labels.size(0),
                10
            ).to(device)

            targets.scatter_(
                1,
                labels.unsqueeze(1),
                1.0
            )

            loss = criterion(
                spk_sum / num_steps,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        end_time = time.time()

        train_losses.append(total_loss)

        epoch_times.append(end_time - start_time)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        test_accuracy = evaluate_snn_nmnist(
            model,
            test_loader,
            device,
            num_steps
        )

        test_accuracies.append(test_accuracy)

        print(f"Test Accuracy: {test_accuracy:.2f}%")

    torch.save(
        model.state_dict(),
        "results/checkpoints/snn_nmnist_T15.pth"
    )

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times,
        "num_steps": num_steps
    }

    with open(
        "results/logs/snn_nmnist_results_T15.json",
        "w"
    ) as f:

        json.dump(results, f)

    return model


def evaluate_snn_nmnist(
    model,
    data_loader,
    device,
    num_steps
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for frames, labels in data_loader:

            frames = frames.float().to(device)
            labels = labels.to(device)

            frames = frames.permute(1,0,2,3,4)

            spk_rec = model(frames)

            spk_sum = spk_rec.sum(dim=0)

            _, predicted = spk_sum.max(1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return 100 * correct / total


if __name__ == "__main__":

    train_snn_nmnist()