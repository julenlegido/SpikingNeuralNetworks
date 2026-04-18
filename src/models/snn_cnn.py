import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN_CNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(128, 10)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, spike_input):

        num_steps = spike_input.size(0)
        batch_size = spike_input.size(1)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []

        for step in range(num_steps):
            x = spike_input[step]

            x = self.pool(self.conv1(x))
            spk1, mem1 = self.lif1(x, mem1)

            x = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(x, mem2)

            x = x.view(batch_size, -1)

            cur3 = self.fc1(x)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)

        return torch.stack(spk_out_rec)