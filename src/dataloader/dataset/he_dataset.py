from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io.image import ImageReadMode, read_image
from torchvision.transforms.functional import resize


class HeDataset(Dataset):
    def __init__(self, directory_path: str):
        self.directory = Path(directory_path)
        self.labels = []
        self.images = []

        self.images = [str(x) for x in self.directory.glob("imagesTr/*.png")]
        self.images.extend([str(x) for x in self.directory.glob("imagesTs/*.png")])
        for image_path in self.images:
            label_path = "_".join(image_path.split("_")[:-1]) + ".png"
            label_path = label_path.replace("imagesTr", "labelsTr")
            label_path = label_path.replace("imagesTs", "labelsTs")
            self.labels.append(label_path)

    def __len__(self):
        return len(self.images)

    def get_mask(self, label):
        return torch.concatenate(
            [
                (label == 0),
                (label == 1),
                (label == 2),
            ]
        ) * 255

    def __getitem__(self, index):
        image = read_image(self.images[index], ImageReadMode.RGB)
        label = read_image(self.labels[index], ImageReadMode.GRAY)
        label = self.get_mask(label)

        resized_image = resize(image, [1080, 1920]).to(torch.uint8)
        resized_label = resize(label, [1080, 1920]).to(torch.uint8)
        return resized_image, resized_label


if __name__ == "__main__":
    from torchvision.io import write_jpeg

    dataset = HeDataset("/Users/babi/Documents/dataset/Dataset110_Tiger")
    i = 0
    for image, mask in dataset:
        write_jpeg(image, "image.jpg")
        write_jpeg(mask[0].unsqueeze(0), "mask.jpg")

        if i == 5:
            break

        i += 1
