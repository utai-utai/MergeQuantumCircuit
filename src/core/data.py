"""
MNIST data loading and the near-identity residual-core initialisation shared by
the classical models (Method, ViT, DiT).  Centralising the transforms keeps the
three experiments consistent; the per-model residual-core nn.Modules stay in
their own experiment files because their forward passes differ.
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src import DATA_DIR


def mnist_transform(pad=0):
    ts = [transforms.ToTensor()]
    if pad:
        ts.append(transforms.Pad(pad, fill=0))            # DiT pads 28->32
    ts.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(ts)


def mnist_dataset(train=True, pad=0, download=True):
    return datasets.MNIST(DATA_DIR, train=train, download=download,
                          transform=mnist_transform(pad))


def mnist_loaders(batch_size=128, pad=0, test_batch=1000,
                  drop_last=False, num_workers=0):
    """(train_loader, test_loader) with a shared transform."""
    train = mnist_dataset(True, pad)
    test = mnist_dataset(False, pad)
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       drop_last=drop_last, num_workers=num_workers),
            DataLoader(test, batch_size=test_batch, shuffle=False))


def mnist_split_loaders(batch_size=256):
    """train_all / digits 0-4 / digits 5-9 / test — for the merging demo (Exp C)."""
    train = mnist_dataset(True)
    test = mnist_dataset(False)
    targets = train.targets
    idx_lo = (targets <= 4).nonzero(as_tuple=True)[0]
    idx_hi = (targets >= 5).nonzero(as_tuple=True)[0]
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(Subset(train, idx_lo), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(train, idx_hi), batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=1000, shuffle=False),
    )


def near_identity_weight(dim, noise=0.01):
    """W = I + small Gaussian noise — keeps the residual core near-unitary (NU)."""
    return torch.eye(dim) + torch.randn(dim, dim) * noise
