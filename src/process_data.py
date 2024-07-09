import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from torchvision.io import encode_png, read_image, ImageReadMode, encode_jpeg
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm
import pandas as pd
import torch


class JobData:
    def __init__(
        self,
        image_path: Path,
        output_directory: Path,
        left_lung_rle: str,
        right_lung_rle: str,
        heart_rle: str,
        height: int,
        width: int,
    ):
        self.image_path: Path = image_path
        self.left_lung_rle: str = left_lung_rle
        self.right_lung_rle: str = right_lung_rle
        self.heart_rle: str = heart_rle
        self.height: int = height
        self.width: int = width
        self.output_directory: Path = output_directory


class CardiacDataProcessor:
    def __init__(
        self,
        path: str,
        output_directory="data/cardiac_processed_dataset",
    ):
        self.input_directory = Path(path)
        self.output_directory = Path(output_directory)
        self.csv = pd.read_csv(
            str(self.input_directory.joinpath("ChestX-Ray8.csv").resolve()),
            engine="pyarrow",
            index_col=0,
        )
        self.images_path = [
            x for x in self.input_directory.glob("chestxray/images_*/**/*.png")
        ]

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
        mask_lung_left = CardiacDataProcessor.rle_to_mask(
            rle_lung_left, height=height, width=width
        )
        mask_lung_right = CardiacDataProcessor.rle_to_mask(
            rle_lung_right, height=height, width=width
        )
        mask_heart = CardiacDataProcessor.rle_to_mask(rle_heart, height=height, width=width)
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

    @staticmethod
    def _process(job: JobData):
        image = read_image(str(job.image_path.resolve()), ImageReadMode.RGB)
        image = CardiacDataProcessor.resize_image(image)

        mask = CardiacDataProcessor.generate_mask(
            job.left_lung_rle,
            job.right_lung_rle,
            job.heart_rle,
            job.height,
            job.width,
        )
        mask = CardiacDataProcessor.resize_image(mask)

        filename = job.image_path.name
        filename_without_extension = filename.split(".")[0]

        new_image_path = job.output_directory.joinpath(
            "image",
            f'{filename_without_extension}.jpg',
        )
        new_label_path = job.output_directory.joinpath("label", filename)

        image_jpeg = encode_jpeg(image)
        label_png = encode_png(mask)

        new_image_path.write_bytes(image_jpeg.numpy())
        new_label_path.write_bytes(label_png.numpy())

    def process(self):
        output_image_path = self.output_directory.joinpath("image")
        output_label_path = self.output_directory.joinpath("label")

        output_image_path.mkdir(parents=True, exist_ok=True)
        output_label_path.mkdir(parents=True, exist_ok=True)

        jobs_list = []
        for image_path in self.images_path:
            filename = image_path.name
            data_row = self.csv.loc[filename]

            job = JobData(
                image_path=image_path,
                left_lung_rle=str(data_row["Left Lung"]),
                right_lung_rle=str(data_row["Right Lung"]),
                heart_rle=str(data_row["Heart"]),
                height=int(data_row["Height"].item()),
                width=int(data_row["Width"].item()),
                output_directory=self.output_directory,
            )
            jobs_list.append(job)

        total_len = len(jobs_list)
        with ProcessPoolExecutor() as executor:
            for _ in tqdm(
                executor.map(self._process, jobs_list),
                total=total_len,
            ):
                pass


# class UavidDatasetProcessor:
#     def __init__(
#         self,
#         path,
#         output_directory="data/processed_dataset",
#         is_train=True,
#     ):
#         directory = Path(path)
#         self.output_directory = Path(output_directory)
#         self.is_train = is_train

#         if not self.output_directory.exists():
#             self.output_directory.mkdir()

#         if self.is_train:
#             self.images = directory.glob("uavid_train/**/Images/*.png")
#             self.labels = directory.glob("uavid_train/**/Labels/*.png")
#         else:
#             self.images = directory.glob("uavid_val/**/Images/*.png")
#             self.labels = directory.glob("uavid_val/**/Labels/*.png")

#         # if len(self.images) is not len(self.labels):
#         #     print("Number of images & label are not the same.")
#         #     return

#     @staticmethod
#     def decode_image(image_path):
#         return read_image(image_path)

#     @staticmethod
#     def encode_image(image):
#         return encode_png(image)

#     @staticmethod
#     def resize_image(image):
#         return resize(
#             image,
#             size=[2160, 3840],
#             interpolation=InterpolationMode.BICUBIC,
#             antialias=True,
#         )

#     @staticmethod
#     def crop_256(image, label):
#         img_array = []
#         label_array = []

#         blocks = [
#             (0, 1024, 0, 2048),
#             (0, 1024, 896, 2944),
#             (0, 1024, 1792, 3840),
#             (568, 1592, 0, 2048),
#             (568, 1592, 896, 2944),
#             (568, 1592, 1792, 3840),
#             (1136, 2160, 0, 2048),
#             (1136, 2160, 896, 2944),
#             (1136, 2160, 1792, 3840),
#         ]
#         for y_min, _, x_min, _ in blocks:
#             for index in range(16):
#                 y, x = index // 8, index % 8
#                 block_y_min = y_min + (y * 256)
#                 block_y_max = y_min + (y + 1) * 256
#                 block_x_min = x_min + x * 256
#                 block_x_max = x_min + (x + 1) * 256
#                 img_array.append(
#                     image[::, block_y_min:block_y_max, block_x_min:block_x_max]
#                 )
#                 label_array.append(
#                     label[::, block_y_min:block_y_max, block_x_min:block_x_max]
#                 )

#         return img_array, label_array

#     @staticmethod
#     def generate_new_name(root, path, number):
#         folder_name = str(root).split("/")[-3]
#         index = path.name.replace(r".png", "")
#         number_string = str(number)
#         return (
#             f"{folder_name}_{index}_0{str(number_string)}.png"
#             if len(number_string) == 1
#             else f"{folder_name}_{index}_{str(number_string)}.png"
#         )

#     @staticmethod
#     def _process(job: JobData):
#         image_path, label_path, output_directory = (
#             job.image_path,
#             job.label,
#             job.output_directory,
#         )

#         image = UavidDatasetProcessor.decode_image(str(image_path))
#         image = UavidDatasetProcessor.resize_image(image)
#         label = UavidDatasetProcessor.decode_image(str(label_path))
#         label = UavidDatasetProcessor.resize_image(label)

#         img_array, label_array = UavidDatasetProcessor.crop_256(
#             image=image, label=label
#         )
#         for index in range(len(img_array)):
#             new_image_path = output_directory.joinpath("image").joinpath(
#                 UavidDatasetProcessor.generate_new_name(
#                     str(image_path.absolute()), image_path, number=index
#                 )
#             )
#             new_label_path = output_directory.joinpath("label").joinpath(
#                 UavidDatasetProcessor.generate_new_name(
#                     str(image_path.absolute()), image_path, number=index
#                 )
#             )
#             jpeg_image = UavidDatasetProcessor.encode_image(img_array[index])
#             jpeg_label = UavidDatasetProcessor.encode_image(label_array[index])

#             new_image_path.write_bytes(jpeg_image.numpy())
#             new_label_path.write_bytes(jpeg_label.numpy())

#     def process(self):
#         output_image_path = (
#             self.output_directory.joinpath("train")
#             if self.is_train
#             else self.output_directory.joinpath("test")
#         )
#         if not output_image_path.exists():
#             output_image_path.mkdir()

#         new_image_path = output_image_path.joinpath("image")
#         new_label_path = output_image_path.joinpath("label")
#         if not new_image_path.exists():
#             new_image_path.mkdir()
#         if not new_label_path.exists():
#             new_label_path.mkdir()

#         images = [x for x in self.images]
#         labels = [x for x in self.labels]
#         jobs_data = [
#             JobData(image, label, output_image_path)
#             for image, label in zip(images, labels)
#         ]

#         total_len = len(jobs_data)
#         with ProcessPoolExecutor() as executor:
#             for index in tqdm(executor.map(self._process, jobs_data)):
#                 pass


def process_images(path: str):
    processor = CardiacDataProcessor(path=path)
    processor.process()
    # train_images_processor = UavidDatasetProcessor(path=path, is_train=True)
    # test_images_processor = UavidDatasetProcessor(path=path, is_train=False)
    # train_images_processor.process()
    # test_images_processor.process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="directory path")
    parsed: argparse.Namespace = parser.parse_args()
    process_images(path=parsed.path)
