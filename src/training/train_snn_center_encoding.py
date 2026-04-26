import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import json

from snntorch import spikegen

from src.data.mnist import get_mnist_dataloaders
from src.models.snn_mlp_rate import SNN_MLP_Rate
from src.utils.device import get_device
from src.utils.center_weight_mask import create_center_weight_mask


def train_snn_center(num_epochs=5, num_steps=25, batch_size=64, lr=1e-3, spike_prob_scale=1.5):
    device = get_device()

    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)

    model = SNN_MLP_Rate().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #create mask
    mask = create_center_weight_mask().to(device)
    mask = mask.unsqueeze(0).unsqueeze(0)  # shape [1,1,28,28]

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

            # APPLY CENTER WEIGHTING
            weighted_images = images * mask

            # apply spike probability scaling
            encoded_images = torch.clamp(weighted_images * spike_prob_scale, 0, 1)

            spike_input = spikegen.rate(encoded_images, num_steps=num_steps)

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

        test_accuracy = evaluate_snn_center(
            model=model,
            data_loader=test_loader,
            device=device,
            num_steps=num_steps,
            spike_prob_scale=spike_prob_scale,
            mask=mask
        )

        test_accuracies.append(test_accuracy)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

    torch.save(
        model.state_dict(),
        f"results/checkpoints/snn_center_model_scale{spike_prob_scale}.pth"
    )

    results = {
        "loss": train_losses,
        "accuracy": test_accuracies,
        "time": epoch_times,
        "spike_prob_scale": spike_prob_scale
    }

    with open(
        f"results/logs/snn_center_results_steps{num_steps}_scale{spike_prob_scale}.json",
        "w"
    ) as f:
        json.dump(results, f)

    return model


def evaluate_snn_center(model, data_loader, device, num_steps, spike_prob_scale, mask):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            weighted_images = images * mask
            encoded_images = torch.clamp(weighted_images * spike_prob_scale, 0, 1)

            spike_input = spikegen.rate(encoded_images, num_steps=num_steps)

            spk_rec = model(spike_input)
            spk_sum = spk_rec.sum(dim=0)

            _, predicted = spk_sum.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    train_snn_center()
