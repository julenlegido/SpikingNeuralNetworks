import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import time
import json

import random
import numpy as np

from src.data.cifar10dvs import get_cifar10_dvs_dataloaders
from src.models.snn_dvs_improved import SNN_DVS
from src.utils.device import get_device


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_frames(frames):

    frames = frames.float()

    max_val = frames.amax(dim=(1, 2, 3, 4), keepdim=True)

    frames = frames / (max_val + 1e-8)

    return frames


def train_snn_cifar10_dvs(
    num_epochs=10,
    num_steps=20,
    batch_size=64,
    lr=1e-4,
    beta=0.95,
    seed=42
):

    set_seed(seed)

    device = get_device()

    train_loader, test_loader = get_cifar10_dvs_dataloaders(
        batch_size=batch_size,
        num_steps=num_steps
    )

    model = SNN_DVS(beta=beta).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    train_losses = []
    test_accuracies = []
    epoch_times = []

    best_accuracy = 0

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0

        start_time = time.time()

        for frames, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        ):

            frames = normalize_frames(frames).to(device)

            labels = labels.to(device)

            frames = frames.permute(1, 0, 2, 3, 4)

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

        test_accuracy = evaluate_snn_cifar10_dvs(
            model=model,
            data_loader=test_loader,
            device=device,
            num_steps=num_steps
        )

        test_accuracies.append(test_accuracy)

        print(f"Test Accuracy: {test_accuracy:.2f}%")

        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            torch.save(
                model.state_dict(),
                f"results/checkpoints/snn_dvs_seed{seed}.pth"
            )

        scheduler.step()

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times,
        "num_steps": num_steps,
        "beta": beta,
        "seed": seed
    }

    with open(
        f"results/logs/snn_dvs_seed{seed}.json",
        "w"
    ) as f:

        json.dump(results, f)

    return model


def evaluate_snn_cifar10_dvs(
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

            frames = normalize_frames(frames).to(device)

            labels = labels.to(device)

            frames = frames.permute(1, 0, 2, 3, 4)

            spk_rec = model(frames)

            spk_sum = spk_rec.sum(dim=0)

            _, predicted = spk_sum.max(1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return 100 * correct / total


if __name__ == "__main__":

    seeds = [42, 123, 999]

    final_accuracies = []

    for seed in seeds:

        print(f"\n========== SEED {seed} ==========\n")

        train_snn_cifar10_dvs(
            seed=seed
        )

        with open(
            f"results/logs/snn_dvs_seed{seed}.json"
        ) as f:

            data = json.load(f)

        final_accuracies.append(
            data["accuracy"][-1]
        )

    mean_acc = np.mean(final_accuracies)

    std_acc = np.std(final_accuracies)

    print("\n========== FINAL RESULTS ==========\n")

    print(f"Mean Accuracy: {mean_acc:.2f}%")

    print(f"Std Accuracy: {std_acc:.2f}")