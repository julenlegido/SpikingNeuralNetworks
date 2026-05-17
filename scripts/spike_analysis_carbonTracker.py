import torch
import json

from tqdm import tqdm
from snntorch import spikegen

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.snn_norm_kaiming import SNN_CNN
from src.utils.device import get_device


def spike_analysis(
    model_path,
    num_steps=15,
    beta=0.3,
    batch_size=64
):

    device = get_device()

    _, test_loader = get_cifar10_dataloaders(
        batch_size=batch_size
    )

    model = SNN_CNN(beta=beta).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    total_spikes = 0
    total_neurons = 0

    with torch.no_grad():

        for images, _ in tqdm(test_loader):

            images = images.to(device)

            spike_input = spikegen.rate(
                images,
                num_steps=num_steps
            )

            spk_rec = model(spike_input)

            total_spikes += spk_rec.sum().item()

            total_neurons += spk_rec.numel()

    firing_rate = total_spikes / total_neurons

    sparsity = 1.0 - firing_rate

    results = {
        "total_spikes": total_spikes,
        "firing_rate": firing_rate,
        "sparsity": sparsity,
        "num_steps": num_steps,
        "beta": beta
    }

    print("\nResults:\n")

    for k, v in results.items():
        print(f"{k}: {v}")

    with open(
        f"results/logs/spike_analysis_beta{beta}_T{num_steps}.json",
        "w"
    ) as f:

        json.dump(results, f)


if __name__ == "__main__":

    spike_analysis(
        model_path="results/checkpoints/snn_MSEKaimingImproved_beta0.3_T15.pth",
        num_steps=15,
        beta=0.3
    )