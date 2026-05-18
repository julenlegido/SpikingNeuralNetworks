import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tqdm import tqdm

from mlxsnn.datasets.nmnist import NMNISTDataset
from mlxsnn import Leaky

num_epochs = 5
batch_size = 64
num_steps = 15
beta = 0.9
lr = 1e-3


train_dataset = NMNISTDataset(
    root="./data/NMNIST",
    train=True,
    num_steps=num_steps
)

test_dataset = NMNISTDataset(
    root="./data/NMNIST",
    train=False,
    num_steps=num_steps
)

class SimpleSNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(34 * 34 * 2, 256)
        self.lif1 = Leaky(beta=beta)

        self.fc2 = nn.Linear(256, 10)
        self.lif2 = Leaky(beta=beta)

    def __call__(self, x):

        batch_size = x.shape[0]

        mem1 = self.lif1.init_state(batch_size, 256)
        mem2 = self.lif2.init_state(batch_size, 10)

        spk_out = []

        for t in range(num_steps):

            xt = x[:, t]

            xt = xt.reshape(batch_size, -1)

            cur1 = self.fc1(xt)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out.append(spk2)

        return mx.stack(spk_out)


model = SimpleSNN()

optimizer = optim.Adam(learning_rate=lr)


def loss_fn(model, x, y):

    spk_out = model(x)

    spk_sum = mx.sum(spk_out, axis=0)

    targets = mx.eye(10)[y]

    loss = mx.mean((spk_sum / num_steps - targets) ** 2)

    return loss


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


for epoch in range(num_epochs):

    start = time.time()

    total_loss = 0

    for i in tqdm(range(0, len(train_dataset), batch_size)):

        batch = [
            train_dataset[j]
            for j in range(i, min(i + batch_size, len(train_dataset)))
        ]

        x = mx.stack([b[0] for b in batch])
        y = mx.array([b[1] for b in batch])

        loss, grads = loss_and_grad_fn(model, x, y)

        optimizer.update(model, grads)

        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()

    end = time.time()

    correct = 0
    total = 0

    for i in range(0, len(test_dataset), batch_size):

        batch = [
            test_dataset[j]
            for j in range(i, min(i + batch_size, len(test_dataset)))
        ]

        x = mx.stack([b[0] for b in batch])
        y = mx.array([b[1] for b in batch])

        spk_out = model(x)

        spk_sum = mx.sum(spk_out, axis=0)

        preds = mx.argmax(spk_sum, axis=1)

        correct += (preds == y).sum().item()

        total += y.shape[0]

    acc = 100 * correct / total

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {total_loss:.4f} | "
        f"Accuracy: {acc:.2f}% | "
        f"Time: {end-start:.2f}s"
    )

total_spikes = 0
total_neurons = 0

for i in range(0, len(test_dataset), batch_size):

    batch = [
        test_dataset[j]
        for j in range(i, min(i + batch_size, len(test_dataset)))
    ]

    x = mx.stack([b[0] for b in batch])

    spk_out = model(x)

    total_spikes += mx.sum(spk_out).item()

    total_neurons += spk_out.size

firing_rate = total_spikes / total_neurons
sparsity = 1 - firing_rate

print("\nSpike Analysis:\n")

print(f"Total spikes: {total_spikes}")
print(f"Firing rate: {firing_rate:.6f}")
print(f"Sparsity: {sparsity:.6f}")

