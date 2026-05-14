import tonic
import tonic.transforms as transforms

from torch.utils.data import DataLoader


def get_nmnist_dataloaders(batch_size=64, num_steps=15):

    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),

        transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=num_steps
        )
    ])

    train_dataset = tonic.datasets.NMNIST(
        save_to="./data",
        train=True,
        transform=frame_transform
    )

    test_dataset = tonic.datasets.NMNIST(
        save_to="./data",
        train=False,
        transform=frame_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, test_loader