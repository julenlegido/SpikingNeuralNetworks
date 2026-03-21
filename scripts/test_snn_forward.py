import torch
from src.data.mnist import get_mnist_dataloaders
from src.models.snn_mlp import SNN_MLP
from src.utils.device import get_device


def main():
    device = get_device()

    # Load data
    train_loader, _ = get_mnist_dataloaders(batch_size=32)

    # Get one batch
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    images = images.to(device)

    # Create model
    model = SNN_MLP().to(device)

    # Forward pass
    output = model(images, num_steps=25)

    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()