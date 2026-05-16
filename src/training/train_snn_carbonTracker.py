import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import json

from snntorch import spikegen
from carbontracker.tracker import CarbonTracker

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.snn_norm_kaiming import SNN_CNN
from src.utils.device import get_device


def train_snn_cifar(
    num_epochs=5,
    num_steps=15,
    batch_size=64,
    lr=1e-3,
    beta=0.95
):

    device = get_device()

    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=batch_size
    )

    model = SNN_CNN(beta=beta).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    # CarbonTracker
    tracker = CarbonTracker(
        epochs=num_epochs,
        log_dir="results/carbontracker/beta_0.5"
    )

    train_losses = []
    test_accuracies = []
    epoch_times = []

    for epoch in range(num_epochs):

        tracker.epoch_start()

        model.train()

        total_loss = 0

        start_time = time.time()

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        ):

            images = images.to(device)
            labels = labels.to(device)

            #RATE ENCODING
            spike_input = spikegen.rate(
                images,
                num_steps=num_steps
            )

            optimizer.zero_grad()

            spk_rec = model(spike_input)

            spk_sum = spk_rec.sum(dim=0)

            # MSE TARGETS
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

        tracker.epoch_end()

        train_losses.append(total_loss)

        epoch_times.append(end_time - start_time)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        test_accuracy = evaluate_snn_cifar(
            model=model,
            data_loader=test_loader,
            device=device,
            num_steps=num_steps
        )

        test_accuracies.append(test_accuracy)

        print(f"Test Accuracy: {test_accuracy:.2f}%")

    tracker.stop()

    torch.save(
        model.state_dict(),
        f"results/checkpoints/snn_MSEKaimingImproved_beta{beta}_T{num_steps}.pth"
    )

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times,
        "num_steps": num_steps,
        "beta": beta
    }

    with open(
        f"results/logs/snn_MSEKaimingImproved_beta{beta}_T{num_steps}.json",
        "w"
    ) as f:

        json.dump(results, f)

    return model


def evaluate_snn_cifar(
    model,
    data_loader,
    device,
    num_steps
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            spike_input = spikegen.rate(
                images,
                num_steps=num_steps
            )

            spk_rec = model(spike_input)

            spk_sum = spk_rec.sum(dim=0)

            _, predicted = spk_sum.max(1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return 100 * correct / total


if __name__ == "__main__":

    train_snn_cifar(
        beta=0.95,
        num_steps=15
    )
