import json


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def estimate_operation_advantage(
    ann_macs,
    snn_firing_rate,
    T,
    mac_energy=3.1,
    sop_energy=0.9,
    label="Experiment"
):
    """
    ann_macs: estimated ANN MAC operations per inference
    snn_firing_rate: measured SNN firing rate
    T: number of time steps (num_steps) used in the SNN simulation
    mac_energy: relative energy cost of MAC (multiplier-accumulator)
    sop_energy: relative energy cost of SOP (synaptic operation/addition)
    """
    snn_active_sops = ann_macs * T * snn_firing_rate

    ann_energy = ann_macs * mac_energy
    snn_energy = snn_active_sops * sop_energy

    operation_ratio = ann_macs / snn_active_sops
    energy_reduction = ann_energy / snn_energy

    print("\n==============================")
    print(label)
    print("==============================")
    print(f"SNN Time Steps (T): {T}")
    print(f"ANN MACs: {ann_macs:,.0f}")
    print(f"SNN firing rate: {snn_firing_rate:.4f}")
    print(f"Estimated active SNN SOPs: {snn_active_sops:,.0f}")
    
    if operation_ratio > 1:
        print(f"Operation reduction: {operation_ratio:.2f}x fewer ops")
    else:
        print(f"Operation multiplier: {1/operation_ratio:.2f}x more raw ops (due to T)")
        
    print(f"Estimated hardware energy reduction: {energy_reduction:.2f}x")

    return {
        "label": label,
        "ann_macs": ann_macs,
        "snn_time_steps": T,
        "snn_firing_rate": snn_firing_rate,
        "estimated_active_snn_sops": snn_active_sops,
        "energy_reduction": energy_reduction,
        "mac_energy": mac_energy,
        "sop_energy": sop_energy
    }


if __name__ == "__main__":

    results = []

    # Load data logs
    cifar_spike = load_json("results/logs/spike_analysis_beta0.5_T15.json")
    nmnist_spike = load_json("results/logs/spike_analysis_nmnist_T10.json")
    dvs_spike = load_json("results/logs/spike_analysis_dvs_T10_beta0.95.json")

    # approximate ANN-equivalent MACs per inference
    cifar_ann_macs = 10_000_000
    nmnist_ann_macs = 2_000_000
    dvs_ann_macs = 10_000_000

    results.append(
        estimate_operation_advantage(
            ann_macs=cifar_ann_macs,
            snn_firing_rate=cifar_spike["firing_rate"],
            T=cifar_spike["num_steps"],
            label="Static CIFAR10 SNN"
        )
    )

    results.append(
        estimate_operation_advantage(
            ann_macs=nmnist_ann_macs,
            snn_firing_rate=nmnist_spike["firing_rate"],
            T=nmnist_spike["num_steps"],
            label="N-MNIST SNN"
        )
    )

    results.append(
        estimate_operation_advantage(
            ann_macs=dvs_ann_macs,
            snn_firing_rate=dvs_spike["firing_rate"],
            T=dvs_spike["num_steps"],
            label="CIFAR10-DVS SNN"
        )
    )

    # Save to file
    with open("results/logs/operation_count_summary.json", "w") as f:
        json.dump(results, f, indent=4)
