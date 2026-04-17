import torch
from tqdm import tqdm
import json
from snntorch import spikegen

from src.data.mnist import get_mnist_dataloaders
from src.models.snn_mlp_rate import SNN_MLP_Rate
from src.utils.device import get_device


def analyze_spikes_rate(num_steps=25, spike_prob_scale=1.5):
    device = get_device()

    _, test_loader = get_mnist_dataloaders(batch_size=64)

    model = SNN_MLP_Rate().to(device)
    model.load_state_dict(torch.load(
        f"results/checkpoints/snn_rate_model_steps{num_steps}_scale{spike_prob_scale}.pth",
        map_location=device
    ))

    model.eval()

    total_spikes = 0
    total_elements = 0

    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            images = images.to(device)

            encoded_images = torch.clamp(images * spike_prob_scale, 0, 1)
            spike_input = spikegen.rate(encoded_images, num_steps=num_steps)

            spk_rec = model(spike_input)

            total_spikes += spk_rec.sum().item()
            total_elements += spk_rec.numel()

    firing_rate = total_spikes / total_elements
    sparsity = 1 - firing_rate

    results = {
        "num_steps": num_steps,
        "spike_prob_scale": spike_prob_scale,
        "total_spikes": total_spikes,
        "firing_rate": firing_rate,
        "sparsity": sparsity
    }

    with open(
        f"results/logs/spike_rate_results_steps{num_steps}_scale{spike_prob_scale}.json",
        "w"
    ) as f:
        json.dump(results, f)

    print(f"\nResults for num_steps={num_steps}, spike_prob_scale={spike_prob_scale}")
    print(f"Total spikes: {total_spikes}")
    print(f"Firing rate: {firing_rate:.6f}")
    print(f"Sparsity: {sparsity:.6f}")


if __name__ == "__main__":
    analyze_spikes_rate()
