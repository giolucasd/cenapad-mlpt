import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_train_val_dataloaders(config: dict):
    data_root = config["dataset"]["data_root"]
    batch_size = config["execution"]["batch_size"]
    num_workers = config["execution"]["num_workers"]
    resolution = config["pipeline"]["resolution"]

    transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )
    val_ds = datasets.MNIST(
        root=data_root,
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
