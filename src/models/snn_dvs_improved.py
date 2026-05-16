import torch
import torch.nn as nn
import snntorch as snn


class SNN_DVS(nn.Module):

    def __init__(self, beta=0.95):

        super().__init__()

        self.beta = beta

        self.conv1 = nn.Conv2d(
            2,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.lif1 = snn.Leaky(beta=beta)

        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(128)

        self.lif2 = snn.Leaky(beta=beta)

        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(256)

        self.lif3 = snn.Leaky(beta=beta)

        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(
            256 * 4 * 4,
            512
        )

        self.lif4 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(
            512,
            10
        )

        self.lif5 = snn.Leaky(beta=beta)

        self._init_weights()

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu"
                )

                nn.init.zeros_(m.bias)

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk5_rec = []

        for step in range(x.size(0)):

            cur = x[step]

            # BLOCK 1
            cur = self.conv1(cur)
            cur = self.bn1(cur)

            spk1, mem1 = self.lif1(cur, mem1)

            cur = self.pool1(spk1)

            # BLOCK 2
            cur = self.conv2(cur)
            cur = self.bn2(cur)

            spk2, mem2 = self.lif2(cur, mem2)

            cur = self.pool2(spk2)

            # BLOCK 3
            cur = self.conv3(cur)
            cur = self.bn3(cur)

            spk3, mem3 = self.lif3(cur, mem3)

            cur = self.pool3(spk3)

            # FC
            cur = cur.view(cur.size(0), -1)

            cur = self.fc1(cur)

            spk4, mem4 = self.lif4(cur, mem4)

            cur = self.fc2(spk4)

            spk5, mem5 = self.lif5(cur, mem5)

            spk5_rec.append(spk5)

        return torch.stack(spk5_rec)