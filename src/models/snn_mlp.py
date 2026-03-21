import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN_MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10, beta=0.9):
        super().__init__()

        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps=25):
        """
        x: [batch, 1, 28, 28]
        """

        # Flatten input
        x = x.view(x.size(0), -1)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Store output spikes over time
        spk2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        # Stack spikes over time: [time, batch, output]
        return torch.stack(spk2_rec)
