import math
import torch
import torch.nn as nn
import snntorch as snn

from src.models.snn_norm import SNNNorm


class SNN_CNN(nn.Module):

    def __init__(
        self,
        beta=0.9,
        threshold=1.0
    ):

        super().__init__()

        self.beta = beta
        self.threshold = threshold

        # ------------------------
        # Conv layers
        # ------------------------

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            padding=1
        )

        self.norm1 = SNNNorm()

        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold
        )

        self.conv2 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        self.norm2 = SNNNorm()

        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=threshold
        )

        self.conv3 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1
        )

        self.norm3 = SNNNorm()

        self.lif3 = snn.Leaky(
            beta=beta,
            threshold=threshold
        )

        self.pool = nn.MaxPool2d(2, 2)

        # ------------------------
        # FC layers
        # ------------------------

        self.fc1 = nn.Linear(
            128 * 4 * 4,
            256
        )

        self.norm4 = SNNNorm()

        self.lif4 = snn.Leaky(
            beta=beta,
            threshold=threshold
        )

        self.fc2 = nn.Linear(
            256,
            10
        )

        self.lif5 = snn.Leaky(
            beta=beta,
            threshold=threshold
        )

        # ------------------------
        # Custom initialization
        # ------------------------

        self._initialize_weights()

    # -------------------------------------------------
    # Threshold-dependent & leak-aware initialization
    # -------------------------------------------------

    def _initialize_weights(self):

        alpha = 1.0 - self.beta

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                n_in = (
                    m.in_channels
                    * m.kernel_size[0]
                    * m.kernel_size[1]
                )

                variance = (
                    2.0
                    * (self.threshold ** 2)
                ) / (n_in * alpha)

                std = math.sqrt(variance)

                nn.init.normal_(
                    m.weight,
                    mean=0.0,
                    std=std
                )

                if m.bias is not None:

                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):

                n_in = m.in_features

                variance = (
                    2.0
                    * (self.threshold ** 2)
                ) / (n_in * alpha)

                std = math.sqrt(variance)

                nn.init.normal_(
                    m.weight,
                    mean=0.0,
                    std=std
                )

                if m.bias is not None:

                    nn.init.zeros_(m.bias)

    # ------------------------
    # Forward
    # ------------------------

    def forward(self, x):

        """
        x shape:
        [num_steps, batch, 3, 32, 32]
        """

        num_steps = x.size(0)

        spk_rec = []

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        for step in range(num_steps):

            cur = x[step]

            # ------------------------
            # Conv block 1
            # ------------------------

            cur = self.conv1(cur)

            cur = self.norm1(cur)

            spk1, mem1 = self.lif1(
                cur,
                mem1
            )

            cur = self.pool(spk1)

            # ------------------------
            # Conv block 2
            # ------------------------

            cur = self.conv2(cur)

            cur = self.norm2(cur)

            spk2, mem2 = self.lif2(
                cur,
                mem2
            )

            cur = self.pool(spk2)

            # ------------------------
            # Conv block 3
            # ------------------------

            cur = self.conv3(cur)

            cur = self.norm3(cur)

            spk3, mem3 = self.lif3(
                cur,
                mem3
            )

            cur = self.pool(spk3)

            # ------------------------
            # Flatten
            # ------------------------

            cur = cur.view(
                cur.size(0),
                -1
            )

            # ------------------------
            # FC1
            # ------------------------

            cur = self.fc1(cur)

            cur = self.norm4(cur)

            spk4, mem4 = self.lif4(
                cur,
                mem4
            )

            # ------------------------
            # FC2
            # ------------------------

            cur = self.fc2(spk4)

            spk5, mem5 = self.lif5(
                cur,
                mem5
            )

            spk_rec.append(spk5)

        return torch.stack(spk_rec)
