import os
import glob

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

import tonic
import tonic.transforms as transforms


class CIFAR10DVSCustom(Dataset):

    classes = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9
    }

    def __init__(self, root="./data/CIFAR10DVS", num_steps=15):

        self.samples = []

        sensor_size = tonic.datasets.CIFAR10DVS.sensor_size

        self.transform = transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=num_steps
        )

        for class_name, label in self.classes.items():

            class_path = os.path.join(root, class_name)

            files = glob.glob(os.path.join(class_path, "*.aedat*"))

            for f in files:
                self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        file_path, label = self.samples[idx]
        
        events = tonic.io.read_aedat4(file_path)
        #events = tonic.io.read_dvs_128(file_path)

        frames = self.transform(events)

        frames = torch.tensor(frames, dtype=torch.float32)

        # resize 128x128 -> 32x32
        frames = F.interpolate(
            frames,
            size=(32, 32),
            mode="nearest"
        )

        return frames, label


def get_cifar10_dvs_dataloaders(
    batch_size=32,
    num_steps=15,
    train_split=0.9
):

    dataset = CIFAR10DVSCustom(
        root="./data/CIFAR10DVS",
        num_steps=num_steps
    )

    print("Dataset length:", len(dataset))

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
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
        shuffle=False
    )
 
    return train_loader, test_loader