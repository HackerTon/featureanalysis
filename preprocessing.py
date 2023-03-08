import glob
import os
from pathlib import Path

import tensorflow as tf


class SeagullDataset:
    @staticmethod
    def get_path(path_dir, is_train=True):
        if is_train:
            trainimg = os.path.join(path_dir, "trainimg", "*.jpg")
            images = glob.glob(trainimg, recursive=True)
            trainmask = os.path.join(path_dir, "trainmask", "*.jpg")
            labels = glob.glob(trainmask, recursive=True)
        else:
            testimg = os.path.join(path_dir, "testimg", "*.jpg")
            images = glob.glob(testimg, recursive=True)
            testmask = os.path.join(path_dir, "testmask", "*.jpg")
            labels = glob.glob(testmask, recursive=True)

        print(len(images), len(labels))

        mask_set = set()
        image_set = set()
        for lbl in labels:
            lbl = lbl.split("\\")[-1]
            mask_set.add(lbl)

        for img in images:
            img = img.split("\\")[-1]
            image_set.add(img)

        complete_path = mask_set.intersection(image_set)
        print(
            f"IMG - LBL NUM: {len(image_set.difference(mask_set))}, Intersection: {len(complete_path)}"
        )

        return [i for i in complete_path]

    @staticmethod
    def get_image_decode(image, label):
        image = tf.io.read_file(image, "image")
        label = tf.io.read_file(label, "label")

        image = tf.image.decode_image(image)
        label = tf.image.decode_image(label)

        return image, label

    @staticmethod
    def path_test(path):
        return (
            r"C:\Alans\seagull\\testimg\\" + path,
            r"C:\Alans\seagull\\testmask\\" + path,
        )

    @staticmethod
    def path_train(path):
        return (
            r"C:\Alans\seagull\trainimg\\" + path,
            r"C:\Alans\seagull\trainmask\\" + path,
        )

    @staticmethod
    def get_mask(image, label):
        labels = []
        labels.append(label[:, :, 0] == 0)
        labels.append(label[:, :, 0] == 255)

        labels = tf.cast(labels, tf.float32)
        image = tf.cast(image, tf.float32)

        # must perform this
        return image, tf.transpose(labels, [1, 2, 0])

    @staticmethod
    def create_ds(batch_size, path_dir="C:\Alans\seagull", ratio=0.8):
        paths = SeagullDataset.get_path(path_dir=path_dir)
        ds_train: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(paths)
        ds_train = ds_train.map(SeagullDataset.path_train, tf.data.AUTOTUNE)

        paths = SeagullDataset.get_path(is_train=False, path_dir=path_dir)
        ds_test: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(paths)
        ds_test = ds_test.map(SeagullDataset.path_test, tf.data.AUTOTUNE)

        ds: tf.data.Dataset = ds_train.concatenate(ds_test)

        ds = ds.shuffle(23124)

        takefortrain = round(23124 * ratio)
        trainds = ds.take(takefortrain)
        testds = ds.skip(takefortrain).take(23124 - takefortrain)

        trainds = trainds.map(SeagullDataset.get_image_decode, tf.data.AUTOTUNE)
        trainds = trainds.map(SeagullDataset.get_mask, tf.data.AUTOTUNE)
        testds = testds.map(SeagullDataset.get_image_decode, tf.data.AUTOTUNE)
        testds = testds.map(SeagullDataset.get_mask, tf.data.AUTOTUNE)

        # # batch and prefetch
        trainds = trainds.batch(batch_size)
        testds = testds.batch(batch_size)

        trainds = trainds.prefetch(tf.data.AUTOTUNE)
        trainds = testds.prefetch(tf.data.AUTOTUNE)

        return trainds, testds


class UavidDataset:
    @staticmethod
    def labels():
        return ["bg", "blding", "road", "tree", "vege", "movcar", "satcar", "human"]

    @staticmethod
    def get_image_decode(image, label):
        image = tf.io.read_file(image, "image")
        label = tf.io.read_file(label, "label")

        image = tf.image.decode_image(image)
        label = tf.image.decode_image(label)

        return image, label

    # [w, h, c], 448, 448, 3
    @staticmethod
    def decode_crop(image, label):
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

        for y_min, y_max, x_min, x_max in blocks:
            for index in range(8):
                y, x = index // 4, index % 4
                block_y_min = y_min + (y * 512)
                block_y_max = y_min + (y + 1) * 512
                block_x_min = x_min + x * 512
                block_x_max = x_min + (x + 1) * 512
                img_array.append(
                    image[block_y_min:block_y_max, block_x_min:block_x_max]
                )
                label_array.append(
                    label[block_y_min:block_y_max, block_x_min:block_x_max]
                )

        return tf.data.Dataset.from_tensor_slices((img_array, label_array))

    @staticmethod
    def get_mask(image, label):
        labels = []
        labels.append(
            (label[:, :, 0] == 0) & (label[:, :, 1] == 0) & (label[:, :, 2] == 0)
        )
        labels.append(
            (label[:, :, 0] == 128) & (label[:, :, 1] == 0) & (label[:, :, 2] == 0)
        )
        labels.append(
            (label[:, :, 0] == 128) & (label[:, :, 1] == 64) & (label[:, :, 2] == 128)
        )
        labels.append(
            (label[:, :, 0] == 0) & (label[:, :, 1] == 128) & (label[:, :, 2] == 0)
        )
        labels.append(
            (label[:, :, 0] == 128) & (label[:, :, 1] == 128) & (label[:, :, 2] == 0)
        )
        labels.append(
            (label[:, :, 0] == 64) & (label[:, :, 1] == 0) & (label[:, :, 2] == 128)
        )
        labels.append(
            (label[:, :, 0] == 192) & (label[:, :, 1] == 0) & (label[:, :, 2] == 192)
        )
        labels.append(
            (label[:, :, 0] == 64) & (label[:, :, 1] == 64) & (label[:, :, 2] == 0)
        )
        labels = tf.cast(labels, tf.float32)
        image = tf.cast(image, tf.float32)

        # must perform this
        return image, tf.transpose(labels, [1, 2, 0])

    @staticmethod
    def create_ds(
        batch_size,
        path_dir="/home/hackerton/Downloads/uavid_v1.5_official_release_image/",
        maximage=False,
        seed=1024,
        test_batch_size=None,
    ):
        directory = Path(path_dir)
        images = [
            str(x.absolute()) for x in directory.glob("uavid_train/**/Images/*.png")
        ]
        labels = [
            str(x.absolute()) for x in directory.glob("uavid_train/**/Labels/*.png")
        ]
        ds_train: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        ds_train = ds_train.shuffle(len(images), seed=seed)

        images = [
            str(x.absolute()) for x in directory.glob("uavid_val/**/Images/*.png")
        ]
        labels = [
            str(x.absolute()) for x in directory.glob("uavid_val/**/Labels/*.png")
        ]
        ds_test: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        ds_train = ds_train.map(UavidDataset.get_image_decode, tf.data.AUTOTUNE)
        ds_test = ds_test.map(UavidDataset.get_image_decode, tf.data.AUTOTUNE)

        if not maximage:
            ds_train = ds_train.flat_map(UavidDataset.decode_crop)
            ds_test = ds_test.flat_map(UavidDataset.decode_crop)

        ds_train = ds_train.map(UavidDataset.get_mask, tf.data.AUTOTUNE)
        ds_test = ds_test.map(UavidDataset.get_mask, tf.data.AUTOTUNE)

        ds_train = ds_train.batch(batch_size)
        ds_test = ds_test.batch(test_batch_size if test_batch_size else batch_size)

        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        return ds_train, ds_test
