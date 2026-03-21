import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataloaders(batch_size=64):
    """
    Returns train and test dataloaders for MNIST.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ])

    train_dataset = datasets.MNIST(
        root="data/raw",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="data/raw",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader