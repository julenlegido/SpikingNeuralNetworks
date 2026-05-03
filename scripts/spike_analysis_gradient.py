import torch
from tqdm import tqdm
import json

from snntorch import spikegen

from src.data.mnist import get_mnist_dataloaders
from src.models.snn_mlp_rate import SNN_MLP_Rate
from src.utils.device import get_device
from src.utils.gradient_encoding import compute_gradient_map, normalize_gradient


def spike_analysis_gradient(num_steps=25, spike_prob_scale=0.5):
    device = get_device()

    _, test_loader = get_mnist_dataloaders(batch_size=64)

    model = SNN_MLP_Rate().to(device)
    model.load_state_dict(torch.load("results/checkpoints/snn_gradient.pth"))
    model.eval()

    total_spikes = 0
    total_neurons = 0

    with torch.no_grad():
        for images, _ in tqdm(test_loader):

            images = images.to(device)

            # 🔥 Gradient encoding
            grad = compute_gradient_map(images)
            grad = normalize_gradient(grad)
            grad = torch.clamp(grad * spike_prob_scale, 0, 1)

            spike_input = spikegen.rate(grad, num_steps=num_steps)

            spk_rec = model(spike_input)

            total_spikes += spk_rec.sum().item()
            total_neurons += spk_rec.numel()

    firing_rate = total_spikes / total_neurons
    sparsity = 1 - firing_rate

    results = {
        "num_steps": num_steps,
        "spike_prob_scale": spike_prob_scale,
        "total_spikes": total_spikes,
        "firing_rate": firing_rate,
        "sparsity": sparsity
    }

    with open(f"results/logs/spike_gradient_results_scale{spike_prob_scale}.json", "w") as f:
        json.dump(results, f)

    print("\nGradient-based encoding results:")
    print(f"Total spikes: {total_spikes}")
    print(f"Firing rate: {firing_rate:.6f}")
    print(f"Sparsity: {sparsity:.6f}")


if __name__ == "__main__":
    spike_analysis_gradient()