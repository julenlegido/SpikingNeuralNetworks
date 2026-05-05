import torch
from tqdm import tqdm
import json

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.snn_norm_kaiming import SNN_CNN
from src.utils.device import get_device
from snntorch import spikegen

# Optional imports (if using custom encodings)
from src.utils.cifar_gradient_encoding import gradient_rate_encoding


def spike_analysis(
    model_path,
    num_steps=10,
    encoding="rate",   # "rate" or "gradient"
    loss="CE",   # "CE" or "MSE"
    scale=1.0,
    batch_size=64
):
    device = get_device()

    _, test_loader = get_cifar10_dataloaders(batch_size=batch_size)

    model = SNN_CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_spikes = 0
    total_neurons = 0

    with torch.no_grad():
        for images, _ in tqdm(test_loader):

            images = images.to(device)

            # 🔁 Choose encoding
            if encoding == "rate":
                spike_input = spikegen.rate(images, num_steps=num_steps)

            elif encoding == "gradient":
                spike_input = gradient_rate_encoding(
                    images, num_steps=num_steps, scale=scale
                )

            else:
                raise ValueError("Unknown encoding")

            spk_rec = model(spike_input)

            # Count spikes
            total_spikes += spk_rec.sum().item()
            total_neurons += spk_rec.numel()

    firing_rate = total_spikes / total_neurons
    sparsity = 1 - firing_rate

    results = {
        "total_spikes": total_spikes,
        "firing_rate": firing_rate,
        "sparsity": sparsity,
        "num_steps": num_steps,
        "encoding": encoding,
        "loss": loss,
        "scale": scale
    }

    print("\nResults:")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Save
    filename = f"results/logs/spike_analysis_{loss}_{encoding}_ts{num_steps}_scale{scale}_gradient.json"
    with open(filename, "w") as f:
        json.dump(results, f)

    return results


if __name__ == "__main__":
    spike_analysis(
        model_path="results/checkpoints/snn_cifar_norm_MSE_ts15_gradientEnc1.5.pth",
        num_steps=15,
        encoding="gradient",
        loss="MSE",
        scale=1.5
    )