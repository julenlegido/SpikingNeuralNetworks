import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import time
import json
import random
import numpy as np

from snntorch import spikegen

from src.data.mnist import get_mnist_dataloaders
from src.models.snn_mlp_rate import SNN_MLP_Rate
from src.utils.device import get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_snn_rate(
    num_epochs=5,
    num_steps=25,
    batch_size=64,
    lr=1e-3,
    spike_prob_scale=0.4,
    seed=42
):
    set_seed(seed)

    device = get_device()

    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size
    )

    model = SNN_MLP_Rate().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    train_losses = []
    test_accuracies = []
    epoch_times = []

    best_accuracy = 0

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0

        start_time = time.time()

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        ):
            images = images.to(device)
            labels = labels.to(device)

            encoded_images = torch.clamp(
                images * spike_prob_scale,
                0,
                1
            )

            # [T,B,1,28,28]
            spike_input = spikegen.rate(
                encoded_images,
                num_steps=num_steps
            )

            optimizer.zero_grad()

            spk_rec = model(spike_input)

            spk_sum = spk_rec.sum(dim=0)

            loss = criterion(
                spk_sum,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        end_time = time.time()

        train_losses.append(total_loss)
        epoch_times.append(
            end_time - start_time
        )

        print(
            f"Epoch {epoch+1}, Loss: {total_loss:.4f}"
        )

        test_accuracy = evaluate_snn_rate(
            model=model,
            data_loader=test_loader,
            device=device,
            num_steps=num_steps,
            spike_prob_scale=spike_prob_scale
        )

        test_accuracies.append(
            test_accuracy
        )

        print(
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            torch.save(
                model.state_dict(),
                f"results/checkpoints/"
                f"snn_rate_experiment_model_seed{seed}_scale{spike_prob_scale}.pth"
            )

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times,
        "num_steps": num_steps,
        "spike_prob_scale": spike_prob_scale,
        "seed": seed,
        "best_accuracy": best_accuracy
    }

    with open(
        f"results/logs/"
        f"snn_rate_experiment_results_seed{seed}_scale{spike_prob_scale}.json",
        "w"
    ) as f:
        json.dump(results, f)

    return model


def evaluate_snn_rate(
    model,
    data_loader,
    device,
    num_steps,
    spike_prob_scale
):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            encoded_images = torch.clamp(
                images * spike_prob_scale,
                0,
                1
            )

            spike_input = spikegen.rate(
                encoded_images,
                num_steps=num_steps
            )

            spk_rec = model(
                spike_input
            )

            spk_sum = spk_rec.sum(dim=0)

            _, predicted = spk_sum.max(1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return 100 * correct / total


if __name__ == "__main__":

    seeds = [42, 123, 999]
    spike_prob_scale=0.4

    final_accuracies = []
    best_accuracies = []

    for seed in seeds:

        print(
            f"\n========== SEED {seed} ==========\n"
        )

        train_snn_rate(
            seed=seed
        )

        with open(
            f"results/logs/"
            f"snn_rate_experiment_results_seed{seed}_scale{spike_prob_scale}.json",
            "r"
        ) as f:
            data = json.load(f)

        final_accuracies.append(
            data["accuracy"][-1]
        )

        best_accuracies.append(
            data["best_accuracy"]
        )

    mean_final = np.mean(
        final_accuracies
    )

    std_final = np.std(
        final_accuracies
    )

    mean_best = np.mean(
        best_accuracies
    )

    std_best = np.std(
        best_accuracies
    )

    summary = {
        "seeds": seeds,
        "final_accuracies": final_accuracies,
        "best_accuracies": best_accuracies,
        "mean_final_accuracy": float(mean_final),
        "std_final_accuracy": float(std_final),
        "mean_best_accuracy": float(mean_best),
        "std_best_accuracy": float(std_best)
    }

    with open(
        "results/logs/"
        "snn_rate_experiment_seed_summary_scale0.4.json",
        "w"
    ) as f:
        json.dump(summary, f)

    print(
        "\n========== FINAL RESULTS ==========\n"
    )

    print(
        f"Final Accuracy Mean: "
        f"{mean_final:.2f}%"
    )

    print(
        f"Final Accuracy Std: "
        f"{std_final:.2f}"
    )

    print(
        f"Best Accuracy Mean: "
        f"{mean_best:.2f}%"
    )

    print(
        f"Best Accuracy Std: "
        f"{std_best:.2f}"
    )