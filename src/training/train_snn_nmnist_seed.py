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

from src.data.nmnist import get_nmnist_dataloaders
from src.models.snn_nmnist import SNN_NMNIST
from src.utils.device import get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_snn_nmnist(
    num_epochs=5,
    num_steps=10,
    batch_size=64,
    lr=1e-3,
    beta=0.9,
    seed=42
):
    set_seed(seed)

    device = get_device()

    train_loader, test_loader = get_nmnist_dataloaders(
        batch_size=batch_size,
        num_steps=num_steps
    )

    model = SNN_NMNIST(beta=beta).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
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
                frames = (frames > 0).float()
                frames = frames.to(device)
                labels = labels.to(device)

                # [B,T,C,H,W] -> [T,B,C,H,W]
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

            test_accuracy = evaluate_snn_nmnist(
                model,
                test_loader,
                device,
                num_steps
            )

            test_accuracies.append(test_accuracy)

            print(f"Test Accuracy: {test_accuracy:.2f}%")

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

                torch.save(
                    model.state_dict(),
                    f"results/checkpoints/snn_nmnist_seed{seed}.pth"
                )

    finally:
        emissions_kg = tracker.stop()
        energy_kwh = tracker._total_energy.kWh if tracker._total_energy else 0.0

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times,
        "num_steps": num_steps,
        "beta": beta,
        "seed": seed,
        "best_accuracy": best_accuracy,
        "energy_consumption_kwh": energy_kwh,  
        "co2_emissions_kg": emissions_kg        
    }

    with open(
        f"results/logs/snn_nmnist_seed{seed}.json",
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
            frames = (frames > 0).float()
            frames = frames.to(device)
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
    best_accuracies = []

    for seed in seeds:
        print(f"\n========== SEED {seed} ==========\n")

        train_snn_nmnist(
            seed=seed
        )

        with open(
            f"results/logs/snn_nmnist_seed{seed}.json",
            "r"
        ) as f:
            data = json.load(f)

        final_accuracies.append(data["accuracy"][-1])
        best_accuracies.append(data["best_accuracy"])

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
        "results/logs/snn_nmnist_seed_summary.json",
        "w"
    ) as f:
        json.dump(summary, f)

    print("\n========== FINAL RESULTS ==========\n")
    print(f"Final Accuracy Mean: {mean_final:.2f}%")
    print(f"Final Accuracy Std: {std_final:.2f}")
    print(f"Best Accuracy Mean: {mean_best:.2f}%")
    print(f"Best Accuracy Std: {std_best:.2f}")
