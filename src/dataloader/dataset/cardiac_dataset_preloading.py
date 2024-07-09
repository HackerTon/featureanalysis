from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class CardiacDatasetPreloading(Dataset):
    dataset_labels = [
        "background",
        "lung",
        "heart",
    ]

    def __init__(self, directory_path: str):
        directory = Path(directory_path)
        self.images = [x for x in directory.joinpath("image").glob("*.jpeg")]
        self.labels = [x for x in directory.joinpath("label").glob("*.png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = read_image(
            str(self.images[index].resolve()),
            ImageReadMode.RGB,
        )
        image = image.float() / 255

        label = read_image(
            str(self.labels[index].resolve()),
            ImageReadMode.RGB,
        )
        label = image.float()
        return image, label
