import torch
import torch.nn as nn
import snntorch as snn


class SNN_DVS_CNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.lif4 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(128, 10)
        self.lif5 = snn.Leaky(beta=beta)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [T, B, 2, 32, 32]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk_rec = []

        for step in range(x.size(0)):
            cur = x[step]

            cur = self.pool(self.conv1(cur))
            spk1, mem1 = self.lif1(cur, mem1)

            cur = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur, mem2)

            cur = self.pool(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur, mem3)

            cur = spk3.view(spk3.size(0), -1)

            cur = self.fc1(cur)
            spk4, mem4 = self.lif4(cur, mem4)

            cur = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur, mem5)

            spk_rec.append(spk5)

        return torch.stack(spk_rec)