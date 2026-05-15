import torch
import torch.nn as nn
import snntorch as snn


class SNN_NMNIST(nn.Module):

    def __init__(self, beta=0.9):

        super().__init__()

        self.conv1 = nn.Conv2d(
            2,
            32,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(2)

        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(2)

        self.lif2 = snn.Leaky(beta=beta)

        self.fc1 = nn.Linear(
            64 * 8 * 8,
            10
        )

        self.lif3 = snn.Leaky(beta=beta)

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu"
                )

            elif isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu"
                )

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []

        for step in range(x.size(0)):

            cur1 = self.pool1(self.conv1(x[step]))

            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool2(self.conv2(spk1))

            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.view(spk2.size(0), -1))

            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)

        return torch.stack(spk3_rec)