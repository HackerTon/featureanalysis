import functools as ft
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import segmentation_models as sm
import tensorflow as tf
import tensorflow.keras as keras


class SeagullDataset:
    @staticmethod
    def get_path(is_train=True, path_dir="C:\Alans\seagull"):
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
    def create_ds(batch_size, ratio=0.8):
        paths = SeagullDataset.get_path()
        ds_train: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(paths)
        ds_train = ds_train.map(SeagullDataset.path_train, tf.data.AUTOTUNE)

        paths = SeagullDataset.get_path(is_train=False)
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
    def get_image_decode(image, label):
        image = tf.io.read_file(image, "image")
        label = tf.io.read_file(label, "label")

        image = tf.image.decode_image(image)
        label = tf.image.decode_image(label)

        return image, label

    # [w, h, c], 448, 448, 3
    @staticmethod
    def decode_crop(image, label):
        image = image[368 // 2 : -(368 // 2), 256 // 2 : -(256 // 2)]
        label = label[368 // 2 : -(368 // 2), 256 // 2 : -(256 // 2)]

        img_array = []
        label_array = []

        for index in range(4 * 8):
            x, y = index // 8, index % 8
            img_array.append(image[448 * x : 448 * (1 + x), 448 * y : 448 * (1 + y)])
            label_array.append(label[448 * x : 448 * (1 + x), 448 * y : 448 * (1 + y)])

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
    def create_ds(batch_size, maximage=False):
        directory = "/home/hackerton/Downloads/uavid_v1.5_official_release/uavid_train/**/Images/*.png"
        images = glob.glob(directory, recursive=True)
        directory = "/home/hackerton/Downloads/uavid_v1.5_official_release/uavid_train/**/Labels/*.png"
        labels = glob.glob(directory, recursive=True)
        ds_train = tf.data.Dataset.from_tensor_slices((images, labels))

        directory = "/home/hackerton/Downloads/uavid_v1.5_official_release/uavid_val/**/Images/*.png"
        images = glob.glob(directory, recursive=True)
        directory = "/home/hackerton/Downloads/uavid_v1.5_official_release/uavid_val/**/Labels/*.png"
        labels = glob.glob(directory, recursive=True)
        ds_test = tf.data.Dataset.from_tensor_slices((images, labels))

        ds_train = ds_train.shuffle(6400)
        ds_train = ds_train.map(UavidDataset.get_image_decode, tf.data.AUTOTUNE)
        ds_test = ds_test.map(UavidDataset.get_image_decode, tf.data.AUTOTUNE)

        if not maximage:
            ds_train = ds_train.flat_map(UavidDataset.decode_crop)
            ds_test = ds_test.flat_map(UavidDataset.decode_crop)

        ds_train = ds_train.map(UavidDataset.get_mask, tf.data.AUTOTUNE)
        ds_test = ds_test.map(UavidDataset.get_mask, tf.data.AUTOTUNE)

        ds_train = ds_train.batch(batch_size)
        ds_test = ds_test.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        return ds_train, ds_test
