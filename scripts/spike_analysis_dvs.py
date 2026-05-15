import torch
import json
from tqdm import tqdm

from src.data.cifar10dvs import get_cifar10_dvs_dataloaders
from src.models.snn_dvs import SNN_DVS_CNN
from src.utils.device import get_device
from src.training.train_snn_dvs import normalize_frames


def spike_analysis_cifar10_dvs(
    model_path="results/checkpoints/snn_dvs_T5.pth",
    num_steps=5,
    beta=0.9
):
    device = get_device()

    _, test_loader = get_cifar10_dvs_dataloaders(
        batch_size=32,
        num_steps=num_steps
    )

    model = SNN_DVS_CNN(beta=beta).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    total_spikes = 0
    total_neurons = 0

    with torch.no_grad():
        for frames, _ in tqdm(test_loader):
            frames = normalize_frames(frames).to(device)
            frames = frames.permute(1, 0, 2, 3, 4)

            spk_rec = model(frames)

            total_spikes += spk_rec.sum().item()
            total_neurons += spk_rec.numel()

    firing_rate = total_spikes / total_neurons
    sparsity = 1 - firing_rate

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
        f"results/logs/spike_analysis_dvs_T{num_steps}.json",
        "w"
    ) as f:
        json.dump(results, f)


if __name__ == "__main__":
    spike_analysis_cifar10_dvs()