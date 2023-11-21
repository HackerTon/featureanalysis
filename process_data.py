from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from torchvision.io import encode_jpeg, read_image
from torchvision.transforms import Resize
from tqdm import tqdm


class JobData:
    def __init__(self, image, label, output_directory):
        self.image: Path = image
        self.label: Path = label
        self.output_directory: Path = output_directory


class DatasetProcessor:
    def __init__(self, path, output_directory="data/processed_dataset", is_train=True):
        directory = Path(path)
        self.output_directory = Path(output_directory)
        self.is_train = is_train

        if not self.output_directory.exists():
            self.output_directory.mkdir()

        if self.is_train:
            self.images = directory.glob("uavid_train/**/Images/*.png")
            self.labels = directory.glob("uavid_train/**/Labels/*.png")
        else:
            self.images = directory.glob("uavid_val/**/Images/*.png")
            self.labels = directory.glob("uavid_val/**/Labels/*.png")

        # if len(self.images) is not len(self.labels):
        #     print("Number of images & label are not the same.")
        #     return

    @staticmethod
    def decode_image(image_path):
        return read_image(image_path)

    @staticmethod
    def encode_image(image):
        return encode_jpeg(image)

    @staticmethod
    def resize_image(image):
        resizer = Resize([2160, 3840], antialias="True")
        return resizer(image)

    @staticmethod
    def crop_256(image, label):
        img_array = []
        label_array = []

        blocks = [
            (0, 1024, 0, 2048),
            (0, 1024, 896, 2944),
            (0, 1024, 1792, 3840),
            (568, 1592, 0, 2048),
            (568, 1592, 896, 2944),
            (568, 1592, 1792, 3840),
            (1136, 2160, 0, 2048),
            (1136, 2160, 896, 2944),
            (1136, 2160, 1792, 3840),
        ]
        for y_min, _, x_min, _ in blocks:
            for index in range(8):
                y, x = index // 4, index % 4
                block_y_min = y_min + (y * 512)
                block_y_max = y_min + (y + 1) * 512
                block_x_min = x_min + x * 512
                block_x_max = x_min + (x + 1) * 512
                img_array.append(
                    image[::, block_y_min:block_y_max, block_x_min:block_x_max]
                )
                label_array.append(
                    label[::, block_y_min:block_y_max, block_x_min:block_x_max]
                )

        return img_array, label_array

    @staticmethod
    def generate_new_name(root, path, number):
        folder_name = str(root).split("/")[-3]
        index = path.name.replace(r".png", "")
        number_string = str(number)
        return (
            f"{folder_name}_{index}_0{str(number_string)}.jpg"
            if len(number_string) == 1
            else f"{folder_name}_{index}_{str(number_string)}.jpg"
        )

    @staticmethod
    def _process(job: JobData):
        image_path, label_path, output_directory = (
            job.image,
            job.label,
            job.output_directory,
        )

        image = DatasetProcessor.decode_image(str(image_path))
        image = DatasetProcessor.resize_image(image)
        label = DatasetProcessor.decode_image(str(label_path))
        label = DatasetProcessor.resize_image(label)

        img_array, label_array = DatasetProcessor.crop_256(image=image, label=label)
        for index in range(len(img_array)):
            new_image_path = output_directory.joinpath("image").joinpath(
                DatasetProcessor.generate_new_name(
                    str(image_path.absolute()), image_path, number=index
                )
            )
            new_label_path = output_directory.joinpath("label").joinpath(
                DatasetProcessor.generate_new_name(
                    str(image_path.absolute()), image_path, number=index
                )
            )
            jpeg_image = DatasetProcessor.encode_image(img_array[index])
            jpeg_label = DatasetProcessor.encode_image(label_array[index])

            new_image_path.write_bytes(jpeg_image.numpy())
            new_label_path.write_bytes(jpeg_label.numpy())

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
        labels = [x for x in self.labels]
        jobs_data = [
            JobData(image, label, output_image_path)
            for image, label in zip(images, labels)
        ]

        total_len = len(jobs_data)
        with ProcessPoolExecutor() as executor:
            for index in tqdm(executor.map(self._process, jobs_data)):
                pass


def process_images(path):
    train_images_processor = DatasetProcessor(path=path, is_train=True)
    test_images_processor = DatasetProcessor(path=path, is_train=False)
    train_images_processor.process()
    test_images_processor.process()


if __name__ == "__main__":
    process_images(path="data/uavid_v1.5_official_release_image/")
