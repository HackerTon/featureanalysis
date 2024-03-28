import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize


class UAVIDDataset4K(Dataset):
    dataset_labels = [
        "background",
        "building",
        "road",
        "tree",
        "vegetation",
        "moving_car",
        "stationary_car",
        "human",
    ]

    def __init__(self, path, is_train=True):
        directory = Path(path)
        if is_train:
            self.images = [
                str(x.absolute()) for x in directory.glob("uavid_train/**/Images/*.png")
            ]
            self.labels = [
                str(x.absolute()) for x in directory.glob("uavid_train/**/Labels/*.png")
            ]
        else:
            self.images = [
                str(x.absolute()) for x in directory.glob("uavid_val/**/Images/*.png")
            ]
            self.labels = [
                str(x.absolute()) for x in directory.glob("uavid_val/**/Labels/*.png")
            ]

        if len(self.images) is not len(self.labels):
            raise Exception("Number of images & label are not the same.")
            return

    def __len__(self):
        return len(self.images)

    @staticmethod
    def decode_image(image_path):
        return read_image(image_path)

    @staticmethod
    def resize_image(image):
        resizer = Resize([2160, 3840], antialias=True)
        return resizer(image)

    @staticmethod
    def label_0and1(label):
        return label.type(torch.float32)

    @staticmethod
    def image_0and1(image):
        return (image / 255).type(torch.float32)

    @staticmethod
    def mask_label(label):
        labels = []
        labels.append((label[0] == 0) & (label[1] == 0) & (label[2] == 0))
        labels.append((label[0] == 128) & (label[1] == 0) & (label[2] == 0))
        labels.append((label[0] == 128) & (label[1] == 64) & (label[2] == 128))
        labels.append((label[0] == 0) & (label[1] == 128) & (label[2] == 0))
        labels.append((label[0] == 128) & (label[1] == 128) & (label[2] == 0))
        labels.append((label[0] == 64) & (label[1] == 0) & (label[2] == 128))
        labels.append((label[0] == 192) & (label[1] == 0) & (label[2] == 192))
        labels.append((label[0] == 64) & (label[1] == 64) & (label[2] == 0))
        return torch.stack(labels)

    def __getitem__(self, index):
        image = self.decode_image(self.images[index])
        image = self.resize_image(image)
        image = self.image_0and1(image)

        label = self.decode_image(self.labels[index])
        label = self.resize_image(label)
        label = self.label_0and1(label)
        label = self.mask_label(label)

        return image, label


class UAVIDDataset(Dataset):
    dataset_labels = [
        "background",
        "building",
        "road",
        "tree",
        "vegetation",
        "moving_car",
        "stationary_car",
        "human",
    ]

    def __init__(self, path, is_train=True):
        directory = Path(path)
        if is_train:
            self.images = [
                str(x.absolute()) for x in directory.glob("train/image/*.png")
            ]
            self.labels = [
                str(x.absolute()) for x in directory.glob("train/label/*.png")
            ]
        else:
            self.images = [
                str(x.absolute()) for x in directory.glob("test/image/*.png")
            ]
            self.labels = [
                str(x.absolute()) for x in directory.glob("test/label/*.png")
            ]

        if len(self.images) != len(self.labels):
            print("Number of images & label are not the same.")
            return

    def __len__(self):
        return len(self.images)

    @staticmethod
    def decode_image(image_path):
        return read_image(image_path)

    @staticmethod
    def label_0and1(label):
        return label.type(torch.float32)

    @staticmethod
    def image_0and1(image):
        return (image / 255).type(torch.float32)

    @staticmethod
    def mask_label(label):
        labels = []
        labels.append((label[0] == 0) & (label[1] == 0) & (label[2] == 0))
        labels.append((label[0] == 128) & (label[1] == 0) & (label[2] == 0))
        labels.append((label[0] == 128) & (label[1] == 64) & (label[2] == 128))
        labels.append((label[0] == 0) & (label[1] == 128) & (label[2] == 0))
        labels.append((label[0] == 128) & (label[1] == 128) & (label[2] == 0))
        labels.append((label[0] == 64) & (label[1] == 0) & (label[2] == 128))
        labels.append((label[0] == 192) & (label[1] == 0) & (label[2] == 192))
        labels.append((label[0] == 64) & (label[1] == 64) & (label[2] == 0))
        return torch.stack(labels)

    def __getitem__(self, index):
        image = self.decode_image(self.images[index])
        image = self.image_0and1(image)
        label = self.decode_image(self.labels[index])
        label = self.mask_label(label)
        label = self.label_0and1(label)
        return image, label


class TextOCRDataset(Dataset):
    def __init__(self, directory, is_train=True):
        if directory == None:
            print("Directory is none")
            return
        self.directory = Path(directory)
        self.images = []
        self.labels = []
        if is_train:
            self.decode(
                file_path=str(
                    self.directory.joinpath("TextOCR_0.1_train.json"),
                )
            )
        else:
            self.decode(
                file_path=str(
                    self.directory.joinpath("TextOCR_0.1_val.json"),
                ),
            )

    @staticmethod
    def decode_image(image_path):
        return read_image(image_path, ImageReadMode.RGB)

    def decode(self, file_path: str):
        validation_label = json.load(open(file_path))
        for image_id, image in validation_label["imgToAnns"].items():
            bounding_box_each_image = []
            for annotation in image:
                annot = validation_label["anns"][f"{annotation}"]
                bounding_box = annot["bbox"]
                x1, y1 = int(bounding_box[0]), int(bounding_box[1])
                x2, y2 = x1 + int(bounding_box[2]), y1 + int(bounding_box[3])
                bounding_box_each_image.append([x1, y1, x2, y2])
            self.images.append(image_id)
            self.labels.append(bounding_box_each_image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.decode_image(
            str(self.directory.joinpath("train_images", f"{self.images[index]}.jpg"))
        )
        mask = torch.zeros([1, image.size(1), image.size(2)], dtype=torch.uint8)

        for bounding_box in self.labels[index]:
            x1, y1, x2, y2 = bounding_box
            _, w, h = mask[..., y1:y2, x1:x2].size()
            mask[..., y1:y2, x1:x2] = torch.tensor([255]).repeat(
                1,
                w,
                h,
            )

        mask = torch.cat([mask, torch.abs(1 - mask)])
        return image, mask


class LungDataset(Dataset):
    def __init__(self, directory, is_train=True):
        if directory == None:
            print("Directory is none")
            return
        self.directory = Path(directory)
        self.images = []
        self.labels = []
        self.area = []
        self.is_train = is_train
        if is_train:
            self.decode(path=self.directory.joinpath("CXR_png"))
        else:
            self.decode(path=self.directory.joinpath("CXR_png"))

    def decode(self, path: Path):
        self.labels = [x for x in self.directory.joinpath("masks").glob("*")]
        for label_image in self.labels:
            filename = label_image.name
            filename_without_extension = filename.split(".")[0].replace("_mask", "")
            self.images.append(path.joinpath(f"{filename_without_extension}.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.decode_image(str(self.images[index]))
        label = self.decode_image_gray(str(self.labels[index]))
        return image, label

    @staticmethod
    def decode_image_gray(image_path):
        return read_image(image_path, ImageReadMode.GRAY)

    @staticmethod
    def decode_image(image_path):
        return read_image(image_path, ImageReadMode.RGB)


class CardiacDataset(Dataset):
    dataset_labels = [
        "background",
        "lung",
        "heart",
    ]

    def __init__(self, directory_path: str):
        directory = Path(directory_path)
        self.images = [str(x.absolute()) for x in directory.glob("train/image/*.png")]
        self.labels = [str(x.absolute()) for x in directory.glob("train/label/*.png")]
        if len(self.images) != len(self.labels):
            raise Exception("Number of images & label are not the same.")

    def __len__(self):
        return len(self.images)

    @staticmethod
    def label_0and1(label):
        return label.float()

    @staticmethod
    def image_0and1(image):
        return (image / 255).float()

    @staticmethod
    def mask_label(label):
        labels = []
        labels.append((label[0] == 0) & (label[1] == 0) & (label[2] == 0))
        labels.append((label[0] == 128) & (label[1] == 0) & (label[2] == 0))
        labels.append((label[0] == 128) & (label[1] == 64) & (label[2] == 128))
        return torch.stack(labels)

    def __getitem__(self, index):
        image = read_image(self.images[index], ImageReadMode.RGB)
        image = self.image_0and1(image)
        label = read_image(self.labels[index], ImageReadMode.RGB)
        label = self.mask_label(label)
        label = self.label_0and1(label)
        return image, label


class CardiacDatasetHDF5(Dataset):
    dataset_labels = [
        "background",
        "lung",
        "heart",
    ]

    def __init__(self, data_path: str, data_path2: str):
        self.data_path = Path(data_path)
        self.data_path2 = Path(data_path2)
        self.dataset_image = None
        self.dataset_label = None
        with h5py.File(str(self.data_path.joinpath("train_image.hdf5")), "r") as file:
            self.dataset_length = len(file["image"])

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        if self.dataset_image is None and self.dataset_label is None:
            self.dataset_image = h5py.File(
                str(self.data_path.joinpath("train_image.hdf5")), "r"
            )["image"]
            self.dataset_label = h5py.File(
                str(self.data_path2.joinpath("train_label.hdf5")), "r"
            )["label"]
        return self.dataset_image[index], self.dataset_label[index]
