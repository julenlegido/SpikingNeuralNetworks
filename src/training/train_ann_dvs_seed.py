import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import time
import json
import random
import numpy as np

from codecarbon import EmissionsTracker

from src.data.cifar10dvs import get_cifar10_dvs_dataloaders
from src.models.ann_event_cnn import ANN_Event_CNN
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

    max_val = frames.amax(
        dim=(1, 2, 3, 4),
        keepdim=True
    )

    return frames / (max_val + 1e-8)


def collapse_time(frames):
    # [B,T,C,H,W] -> [B,C,H,W]

    frames = normalize_frames(frames)

    return frames.sum(dim=1)


def train_ann_dvs(
    num_epochs=5,
    num_steps=10,
    batch_size=64,
    lr=1e-3,
    seed=42
):

    set_seed(seed)

    device = get_device()

    train_loader, test_loader = get_cifar10_dvs_dataloaders(
        batch_size=batch_size,
        num_steps=num_steps
    )

    model = ANN_Event_CNN(
        input_channels=2,
        input_size=32
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    train_losses = []
    test_accuracies = []
    epoch_times = []

    best_accuracy = 0

    output_dir = "results/codecarbon"
    os.makedirs(output_dir, exist_ok=True)

    tracker = EmissionsTracker(output_dir=output_dir)
    tracker.start()

    try:
        for epoch in range(num_epochs):

            model.train()

            total_loss = 0

            start_time = time.time()

            for frames, labels in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}"
            ):

                frames = collapse_time(frames).to(device)

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(frames)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            end_time = time.time()

            epoch_times.append(end_time - start_time)

            train_losses.append(total_loss)

            acc = evaluate_ann_dvs(
                model,
                test_loader,
                device
            )

            test_accuracies.append(acc)

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

            print(f"Test Accuracy: {acc:.2f}%")

            if acc > best_accuracy:

                best_accuracy = acc

                torch.save(
                    model.state_dict(),
                    f"results/checkpoints/ann_dvs_seed{seed}.pth"
                )

    finally:
        emissions_kg = tracker.stop()
        energy_kwh = tracker._total_energy.kWh if tracker._total_energy else 0.0

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times,
        "num_steps": num_steps,
        "seed": seed,
        "best_accuracy": best_accuracy,
        "energy_consumption_kwh": energy_kwh,  
        "co2_emissions_kg": emissions_kg        
    }

    with open(
        f"results/logs/ann_dvs_seed{seed}.json",
        "w"
    ) as f:

        json.dump(results, f)

    return model


def evaluate_ann_dvs(
    model,
    data_loader,
    device
):

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

            correct += (
                predicted == labels
            ).sum().item()

    return 100 * correct / total


if __name__ == "__main__":

    seeds = [42, 123, 999]

    final_accuracies = []
    best_accuracies = []

    for seed in seeds:

        print(f"\n========== SEED {seed} ==========\n")

        train_ann_dvs(
            seed=seed
        )

        with open(
            f"results/logs/ann_dvs_seed{seed}.json",
            "r"
        ) as f:

            data = json.load(f)

        final_accuracies.append(
            data["accuracy"][-1]
        )

        best_accuracies.append(
            data["best_accuracy"]
        )

    mean_final = np.mean(final_accuracies)

    std_final = np.std(final_accuracies)

    mean_best = np.mean(best_accuracies)

    std_best = np.std(best_accuracies)

    summary = {
        "seeds": seeds,
        "final_accuracies": final_accuracies,
        "best_accuracies": best_accuracies,
        "mean_final_accuracy": mean_final,
        "std_final_accuracy": std_final,
        "mean_best_accuracy": mean_best,
        "std_best_accuracy": std_best
    }

    with open(
        "results/logs/ann_dvs_seed_summary.json",
        "w"
    ) as f:

        json.dump(summary, f)

    print("\n========== FINAL RESULTS ==========\n")

    print(f"Final Accuracy Mean: {mean_final:.2f}%")

    print(f"Final Accuracy Std: {std_final:.2f}")

    print(f"Best Accuracy Mean: {mean_best:.2f}%")

    print(f"Best Accuracy Std: {std_best:.2f}")
