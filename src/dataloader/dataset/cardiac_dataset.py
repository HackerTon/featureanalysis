from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import InterpolationMode, resize


class CardiacDataset(Dataset):
    dataset_labels = [
        "background",
        "lung",
        "heart",
    ]

    def __init__(self, directory_path: str):
        directory = Path(directory_path)
        self.csv = pd.read_csv(
            str(directory.joinpath("ChestX-Ray8.csv").resolve()),
            engine="pyarrow",
            index_col=0,
        )
        self.images = [x for x in directory.glob("chestxray/images_*/**/*.png")]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def resize_image(image):
        return resize(
            image,
            size=[512, 512],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    @staticmethod
    def rle_to_mask(rle: str, height: int, width: int):
        runs = torch.tensor([int(x) for x in rle.split()])
        starts = runs[::2]
        lengths = runs[1::2]
        mask = torch.zeros([height * width], dtype=torch.uint8)

        for start, lengths in zip(starts, lengths):
            start -= 1
            end = start + lengths
            mask[start:end] = 255
        return mask.reshape((height, width))

    @staticmethod
    def generate_mask(rle_lung_left, rle_lung_right, rle_heart, height, width):
        mask_lung_left = CardiacDataset.rle_to_mask(
            rle_lung_left, height=height, width=width
        )
        mask_lung_right = CardiacDataset.rle_to_mask(
            rle_lung_right, height=height, width=width
        )
        mask_heart = CardiacDataset.rle_to_mask(rle_heart, height=height, width=width)
        mask_lung = torch.clamp(
            (mask_lung_left + mask_lung_right),
            min=0,
            max=255,
        )
        mask_background = torch.clamp(
            255 - (mask_heart + mask_lung),
            min=0,
            max=255,
        )
        return torch.stack([mask_background, mask_lung, mask_heart])

    def __getitem__(self, index):
        filename = self.images[index].name
        image = read_image(
            str(self.images[index].resolve()),
            ImageReadMode.RGB,
        )
        image = CardiacDataset.resize_image(image)
        data_row = self.csv.loc[filename]
        mask = CardiacDataset.generate_mask(
            data_row["Left Lung"],
            data_row["Right Lung"],
            data_row["Heart"],
            data_row["Height"],
            data_row["Width"],
        )
        mask = CardiacDataset.resize_image(mask)
        return image, mask
    

# if __name__ == '__main__':
#     for image, label in CardiacDataset('/pool/storage/dataset/chest'):
#         print(label.min(), label.max())
#         print(image.min(), image.max())
#         break