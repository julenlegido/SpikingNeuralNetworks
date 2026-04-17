import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN_MLP_Rate(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10, beta=0.9):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, spike_input):
        """
        spike_input shape: [num_steps, batch, 1, 28, 28]
        """

        num_steps = spike_input.size(0)
        batch_size = spike_input.size(1)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        for step in range(num_steps):
            x = spike_input[step]
            x = x.view(batch_size, -1)

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)
