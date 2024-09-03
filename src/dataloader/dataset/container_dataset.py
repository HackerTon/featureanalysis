import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, crop
from torchvision.io.image import ImageReadMode, read_image


class ContainerDataset(Dataset):
    def __init__(self, directory_path):
        self.directory = Path(directory_path)
        self.image_label = []
        self.decode(
            file_path=str(self.directory.joinpath("train/_annotations.coco.json")),
            type="train",
        )

    def decode(self, file_path: str, type: str):
        with open(file_path) as file:
            jsonData = json.load(file)
            for image in jsonData["images"]:
                image_id = image["id"]
                image_filename = image["file_name"]
                image_height = image["height"]
                image_width = image["width"]
                for annotation in jsonData["annotations"]:
                    if annotation["image_id"] == image_id:
                        bounding_box = annotation["bbox"]
                        x1, y1 = int(bounding_box[0]), int(bounding_box[1])
                        x2, y2 = x1 + int(bounding_box[2]), y1 + int(bounding_box[3])
                        self.image_label.append(
                            {
                                "image_filename": f"{type}/{image_filename}",
                                "bbox": [x1, y1, x2, y2],
                            }
                        )

    def __len__(self):
        return len(self.image_label)

    def decode_image(self, image_path):
        return read_image(image_path, ImageReadMode.RGB)

    def generate_mask(
        self,
        x: int,
        y: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int,
    ) -> torch.Tensor:
        foreground_mask = torch.zeros(
            [image_height, image_width],
            dtype=torch.uint8,
        )
        foreground_mask[y:y2, x:x2] = 255
        background_mask = torch.abs(255 - foreground_mask)
        background_mask = resize(
            background_mask.unsqueeze(0),
            size=[1080, 1920],
        )
        foreground_mask = resize(
            foreground_mask.unsqueeze(0),
            size=[1080, 1920],
        )
        return torch.concatenate([background_mask, foreground_mask])

    def __getitem__(self, index):
        image_path = f"{self.image_label[index]['image_filename']}"
        image = self.decode_image(str(self.directory.joinpath(image_path)))
        x1, y1, x2, y2 = self.image_label[index]["bbox"]
        mask = self.generate_mask(
            x=x1,
            y=y1,
            x2=x2,
            y2=y2,
            image_height=image.shape[1],
            image_width=image.shape[2],
        )
        resized_image = resize(image, [1080, 1920])
        return resized_image, mask


if __name__ == "__main__":
    from torchvision.io import write_jpeg

    dataset = ContainerDataset(
        directory_path="/pool/labelstudio/dataset/container_dataset_2"
    )
    for image, mask in dataset:
        print(image.shape)
        print(mask.shape)
        write_jpeg(image, "image.jpg")
        write_jpeg(mask[1].unsqueeze(0), "mask.jpg")
        break
