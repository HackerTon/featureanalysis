from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class CardiacDatasetPreloading(Dataset):
    dataset_labels = [
        "background",
        "lung",
        "heart",
    ]

    def __init__(self, directory_path: str):
        self.directory = Path(directory_path)
        self.images = [x for x in self.directory.joinpath("image").glob("*.jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = read_image(
            str(self.images[index].resolve()),
            ImageReadMode.RGB,
        )
        label_path = str(self.directory.joinpath('label', f'{self.images[index].name.split('.')[0]}.png').resolve())
        label = read_image(
            str(label_path),
            ImageReadMode.RGB,
        )
        return image, label
