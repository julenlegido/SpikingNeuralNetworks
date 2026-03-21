import torch
from tqdm import tqdm

from src.data.mnist import get_mnist_dataloaders
from src.models.snn_mlp import SNN_MLP
from src.utils.device import get_device
from src.evaluation.spike_metrics import compute_spike_stats


def analyze_spikes(num_steps=100):
    device = get_device()

    train_loader, _ = get_mnist_dataloaders(batch_size=64)

    model = SNN_MLP().to(device)

    # load trained model
    model.load_state_dict(torch.load(
        f"results/checkpoints/snn_model_{num_steps}_steps.pth",
        map_location=device
    ))

    model.eval()

    total_spikes = 0
    total_elements = 0

    with torch.no_grad():
        for images, _ in tqdm(train_loader):

            images = images.to(device)

            spk_rec = model(images, num_steps=num_steps)

            stats = compute_spike_stats(spk_rec)

            total_spikes += stats["total_spikes"]
            total_elements += spk_rec.numel()

    firing_rate = total_spikes / total_elements
    sparsity = 1 - firing_rate

    print(f"\nResults for {num_steps} timesteps:")
    print(f"Total spikes: {total_spikes}")
    print(f"Firing rate: {firing_rate:.6f}")
    print(f"Sparsity: {sparsity:.6f}")


if __name__ == "__main__":
    analyze_spikes()
