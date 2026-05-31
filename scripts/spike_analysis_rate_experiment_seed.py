import torch
import numpy as np
import json

from tqdm import tqdm

from snntorch import spikegen

from src.data.mnist import get_mnist_dataloaders
from src.models.snn_mlp_rate import SNN_MLP_Rate
from src.utils.device import get_device
from src.evaluation.spike_metrics import compute_spike_stats


def analyze_model(
    checkpoint_path,
    scale,
    num_steps=25
):
    device = get_device()

    test_loader, _ = get_mnist_dataloaders(
        batch_size=64
    )

    model = SNN_MLP_Rate().to(device)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.eval()

    total_spikes = 0
    total_elements = 0

    with torch.no_grad():

        for images, _ in tqdm(
            test_loader,
            leave=False
        ):

            images = images.to(device)

            encoded_images = torch.clamp(
                images * scale,
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

            stats = compute_spike_stats(
                spk_rec
            )

            total_spikes += (
                stats["total_spikes"]
            )

            total_elements += (
                spk_rec.numel()
            )

    firing_rate = (
        total_spikes / total_elements
    )

    sparsity = (
        1 - firing_rate
    )

    return {
        "total_spikes": float(total_spikes),
        "firing_rate": float(firing_rate),
        "sparsity": float(sparsity)
    }


if __name__ == "__main__":

    scales = [
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.5
    ]

    seeds = [
        42,
        123,
        999
    ]

    num_steps = 25

    for scale in scales:

        print(
            f"\n========== SCALE {scale} ==========\n"
        )

        firing_rates = []
        sparsities = []
        total_spikes_list = []

        per_seed_results = []

        for seed in seeds:

            checkpoint = (
                f"results/checkpoints/"
                f"snn_rate_experiment_model_"
                f"seed{seed}_scale{scale}.pth"
            )

            print(
                f"Analyzing seed {seed}"
            )

            results = analyze_model(
                checkpoint_path=checkpoint,
                scale=scale,
                num_steps=num_steps
            )

            firing_rates.append(
                results["firing_rate"]
            )

            sparsities.append(
                results["sparsity"]
            )

            total_spikes_list.append(
                results["total_spikes"]
            )

            per_seed_results.append({
                "seed": seed,
                **results
            })

        summary = {

            "scale": scale,

            "per_seed_results":
                per_seed_results,

            "mean_firing_rate":
                float(
                    np.mean(
                        firing_rates
                    )
                ),

            "std_firing_rate":
                float(
                    np.std(
                        firing_rates
                    )
                ),

            "mean_sparsity":
                float(
                    np.mean(
                        sparsities
                    )
                ),

            "std_sparsity":
                float(
                    np.std(
                        sparsities
                    )
                ),

            "mean_total_spikes":
                float(
                    np.mean(
                        total_spikes_list
                    )
                ),

            "std_total_spikes":
                float(
                    np.std(
                        total_spikes_list
                    )
                )
        }

        output_file = (
            f"results/logs/"
            f"spike_summary_scale{scale}.json"
        )

        with open(
            output_file,
            "w"
        ) as f:

            json.dump(
                summary,
                f,
                indent=4
            )

        print(
            f"\nResults for scale {scale}"
        )

        print(
            f"Mean Firing Rate: "
            f"{summary['mean_firing_rate']:.6f}"
        )

        print(
            f"Std Firing Rate: "
            f"{summary['std_firing_rate']:.6f}"
        )

        print(
            f"Mean Sparsity: "
            f"{summary['mean_sparsity']:.6f}"
        )

        print(
            f"Std Sparsity: "
            f"{summary['std_sparsity']:.6f}"
        )

        print(
            f"Mean Total Spikes: "
            f"{summary['mean_total_spikes']:.0f}"
        )

        print(
            f"Std Total Spikes: "
            f"{summary['std_total_spikes']:.0f}"
        )

        print(
            f"\nSaved to:"
        )

        print(
            output_file
        )

    print(
        "\n========== FINISHED ==========\n"
    )
