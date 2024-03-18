import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image, write_png
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm


class JobData:
    def __init__(
        self,
        image,
        mask_lung_left,
        mask_lung_right,
        mask_heart,
        height,
        width,
        output_directory,
    ):
        self.image = image
        self.rle_lung_left = mask_lung_left
        self.rle_lung_right = mask_lung_right
        self.rle_heart = mask_heart
        self.height = height
        self.width = width
        self.output_directory: Path = output_directory


class DatasetProcessor:
    def __init__(self, path, output_directory="data/cardiac", is_train=True):
        directory = Path(path)
        self.output_directory = Path(output_directory)
        self.is_train = is_train

        if not self.output_directory.exists():
            self.output_directory.mkdir()

        self.csv = pd.read_csv(
            str(directory.joinpath("chestxray.csv")),
            engine="pyarrow",
            index_col=0,
        )
        self.images = directory.glob("chestxray/images_*/**/*.png")

    @staticmethod
    def resize_image(image):
        return resize(
            image,
            size=[512, 512],
            interpolation=InterpolationMode.NEAREST,
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
        output_image = torch.zeros([height, width, 3], dtype=torch.uint8)
        mask_lung_left = DatasetProcessor.rle_to_mask(
            rle_lung_left, height=height, width=width
        )
        mask_lung_right = DatasetProcessor.rle_to_mask(
            rle_lung_right, height=height, width=width
        )
        mask_heart = DatasetProcessor.rle_to_mask(rle_heart, height=height, width=width)
        output_image[((mask_lung_left + mask_lung_right) == 255)] = torch.tensor(
            [128, 0, 0],
            dtype=torch.uint8,
        )
        output_image[mask_heart == 255] = torch.tensor(
            [128, 64, 128],
            dtype=torch.uint8,
        )
        return output_image.permute([2, 0, 1])

    @staticmethod
    def generate_new_name(root: Path, path):
        filename = path.split("/")[-1]
        return str(root.joinpath(filename).absolute())

    @staticmethod
    def _process(job: JobData):
        (
            image_path,
            rle_lung_left,
            rle_lung_right,
            rle_heart,
            height,
            width,
            output_directory,
        ) = (
            job.image,
            job.rle_lung_left,
            job.rle_lung_right,
            job.rle_heart,
            job.height,
            job.width,
            job.output_directory,
        )

        image = read_image(image_path, ImageReadMode.RGB)
        mask = DatasetProcessor.generate_mask(
            rle_lung_left,
            rle_lung_right,
            rle_heart,
            height,
            width,
        )

        image = DatasetProcessor.resize_image(image)
        mask = DatasetProcessor.resize_image(mask)
        new_image_path = DatasetProcessor.generate_new_name(
            output_directory.joinpath("image"),
            image_path,
        )
        new_label_path = DatasetProcessor.generate_new_name(
            output_directory.joinpath("label"),
            image_path,
        )
        write_png(image.cpu(), new_image_path)
        write_png(mask.cpu(), new_label_path)

    def process(self):
        output_image_path = (
            self.output_directory.joinpath("train")
            if self.is_train
            else self.output_directory.joinpath("test")
        )
        if not output_image_path.exists():
            output_image_path.mkdir()

        new_image_path = output_image_path.joinpath("image")
        new_label_path = output_image_path.joinpath("label")
        if not new_image_path.exists():
            new_image_path.mkdir()
        if not new_label_path.exists():
            new_label_path.mkdir()

        images = [x for x in self.images]
        jobs_data = []
        for image_path in images:
            image_name = image_path.name
            selected_row = self.csv.loc[image_name]
            left_lung_rle = selected_row["Left Lung"]
            right_lung_rle = selected_row["Right Lung"]
            heart_rle = selected_row["Heart"]
            height = selected_row["Height"]
            width = selected_row["Width"]
            jobs_data.append(
                JobData(
                    image=str(image_path),
                    mask_lung_left=left_lung_rle,
                    mask_lung_right=right_lung_rle,
                    mask_heart=heart_rle,
                    height=height,
                    width=width,
                    output_directory=output_image_path,
                )
            )

        print("Start Generating")
        with ProcessPoolExecutor(max_workers=1) as executor:
            for _ in tqdm(executor.map(self._process, jobs_data), total=len(jobs_data)):
                pass

    def process_hdf5(self, output_directory: str):
        output_image_path = Path(output_directory)
        if not output_image_path.exists():
            output_image_path.mkdir()

        images = [x for x in self.images]
        jobs_data = []
        for image_path in images:
            image_name = image_path.name
            selected_row = self.csv.loc[image_name]
            left_lung_rle = selected_row["Left Lung"]
            right_lung_rle = selected_row["Right Lung"]
            heart_rle = selected_row["Heart"]
            height = selected_row["Height"]
            width = selected_row["Width"]
            jobs_data.append(
                JobData(
                    image=str(image_path),
                    mask_lung_left=left_lung_rle,
                    mask_lung_right=right_lung_rle,
                    mask_heart=heart_rle,
                    height=height,
                    width=width,
                    output_directory=None,
                )
            )

        print("Start Generating")
        hdf5_file = h5py.File(
            str(output_image_path.joinpath("train.hdf5")),
            "w",
        )

        dataset_images = hdf5_file.create_dataset(
            "image",
            shape=(len(jobs_data), 3, 512, 512),
            dtype="i1",
        )
        dataset_labels = hdf5_file.create_dataset(
            "label",
            shape=(len(jobs_data), 3, 512, 512),
            dtype="i1",
        )

        for idx, job in enumerate(tqdm(jobs_data, total=len(jobs_data))):
            (
                image_path,
                rle_lung_left,
                rle_lung_right,
                rle_heart,
                height,
                width,
            ) = (
                job.image,
                job.rle_lung_left,
                job.rle_lung_right,
                job.rle_heart,
                job.height,
                job.width,
            )

            image = read_image(image_path, ImageReadMode.RGB)
            mask = DatasetProcessor.generate_mask(
                rle_lung_left,
                rle_lung_right,
                rle_heart,
                height,
                width,
            )

            image = DatasetProcessor.resize_image(image)
            mask = DatasetProcessor.resize_image(mask)
            dataset_images[idx] = image
            dataset_labels[idx] = mask

        hdf5_file.close()


def process_images(path, output_path):
    train_images_processor = DatasetProcessor(path=path, is_train=True)
    # test_images_processor = DatasetProcessor(path=path, is_train=False)
    # train_images_processor.process()
    train_images_processor.process_hdf5(output_directory=output_path)
    # test_images_processor.process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path to uavid directory")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parsed: argparse.Namespace = parser.parse_args()
    process_images(path=parsed.path, output_path=parsed.output)
