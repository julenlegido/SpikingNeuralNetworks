import torch
import torch.nn as nn
import snntorch as snn

from src.models.snn_norm import SNNNorm


class SNN_CNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.norm1 = SNNNorm()
        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = SNNNorm()
        self.lif2 = snn.Leaky(beta=beta)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.norm3 = SNNNorm()
        self.lif3 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(128, 10)
        self.lif4 = snn.Leaky(beta=beta)

    def forward(self, x):
        """
        x shape: [num_steps, batch, 3, 32, 32]
        """

        num_steps = x.size(0)

        spk_rec = []

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        for step in range(num_steps):

            cur = x[step]

            # Conv 1
            cur = self.conv1(cur)
            cur = self.norm1(cur)
            spk1, mem1 = self.lif1(cur, mem1)

            cur = self.pool(spk1)

            # Conv 2
            cur = self.conv2(cur)
            cur = self.norm2(cur)
            spk2, mem2 = self.lif2(cur, mem2)

            cur = self.pool(spk2)

            # Flatten
            cur = cur.view(cur.size(0), -1)

            # FC 1
            cur = self.fc1(cur)
            cur = self.norm3(cur)
            spk3, mem3 = self.lif3(cur, mem3)

            # Output
            cur = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur, mem4)

            spk_rec.append(spk4)

        return torch.stack(spk_rec)