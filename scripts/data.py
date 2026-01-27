"""
Download MNIST data.

Usage
-----
    uv run scripts/data.py --data-root /path/to/data
"""

import argparse
from pathlib import Path

from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Download the MNIST dataset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset location; relative paths are resolved inside the repository.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    print("MNIST downloaded successfully.")


if __name__ == "__main__":
    main()
