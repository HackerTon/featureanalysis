from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io.image import ImageReadMode, read_image


class HeDataset(Dataset):
    def __init__(self, directory_path: str):
        self.directory = Path(directory_path)
        self._initialize_dataset()

    def _initialize_dataset(self):
        image_length = len(list(self.directory.glob("*_image.png")))
        label_length = len(list(self.directory.glob("*_label.png")))

        if image_length != label_length:
            raise Exception(
                f"Image length {image_length} != label length {label_length}"
            )

        self.dataset_length = image_length

    def __len__(self):
        return self.dataset_length

    def get_mask(self, label):
        return (
            torch.concatenate(
                [
                    (label == 0),  # Background class
                    (label == 1),  # Stroma class
                    (label == 2),  # Apa class
                ]
            )
            * 255
        )

    def __getitem__(self, index):
        image_path = str(self.directory.joinpath(f"{index}_image.png"))
        label_path = str(self.directory.joinpath(f"{index}_label.png"))
        image = read_image(image_path, ImageReadMode.RGB)
        label = read_image(label_path, ImageReadMode.GRAY)
        label = self.get_mask(label).to(torch.uint8)
        return image, label


if __name__ == "__main__":
    from torchvision.io import write_jpeg
    from torch.utils.data import DataLoader

    dataset = HeDataset("data/ukmtils_pseudo")
    i = 0
    for image, mask in DataLoader(dataset, shuffle=False, num_workers=4):
        # write_jpeg(image, "image.jpg")
        # write_jpeg(mask[0].unsqueeze(0), "mask.jpg")
        # if i == 5:
        #     break

        i += 1
        break
