import torch


def compute_spike_stats(spk_rec):
    """
    spk_rec shape: [time, batch, neurons]
    """

    total_spikes = spk_rec.sum().item()

    # total number of elements
    total_elements = spk_rec.numel()

    # firing rate (how often neurons fire)
    firing_rate = total_spikes / total_elements

    # sparsity = how many are zero
    sparsity = 1 - firing_rate

    return {
        "total_spikes": total_spikes,
        "firing_rate": firing_rate,
        "sparsity": sparsity
    }
