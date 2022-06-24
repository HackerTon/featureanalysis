import datetime
import functools as ft
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras
from django.apps import AppConfig
from django.conf import settings
from base64 import b64encode

# disable GPU computation
tf.config.set_visible_devices([], "GPU")
sm.set_framework("tf.keras")
sm.framework()

tf.random.set_seed(1024)
SEED = 100


def create_backbone_efficient():
    _backbone = keras.applications.EfficientNetB0(include_top=False)

    outputs = [
        layer.output
        for layer in _backbone.layers
        if layer.name
        in [
            "block2a_activation",
            "block3a_activation",
            "block5a_activation",
            "block7a_activation",
        ]
    ]

    return tf.keras.Model(
        inputs=[_backbone.input], outputs=outputs, name="efficientb0_backbone"
    )


class FPN(tf.keras.layers.Layer):
    def __init__(self, backbone=None, **kwargs):
        super().__init__(name="Feature_Pyramid_Network", **kwargs)

        self.backbone = create_backbone_efficient()

        self.conv5_1x1 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(1, 1), padding="same"
        )
        self.conv4_1x1 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(1, 1), padding="same"
        )
        self.conv3_1x1 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(1, 1), padding="same"
        )
        self.conv2_1x1 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(1, 1), padding="same"
        )
        self.conv5_3x3_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv5_3x3_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv4_3x3_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv4_3x3_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv3_3x3_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv3_3x3_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv2_3x3_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv2_3x3_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.upscale = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, images, training=False):
        # 112x112, 56x56, 28x28, 14x14
        conv2, conv3, conv4, conv5 = self.backbone(images, training=False)
        conv5_m = self.conv5_1x1(conv5)
        conv5_p = self.conv5_3x3_1(conv5_m)
        conv5_p = self.conv5_3x3_2(conv5_p)

        conv4_m_1 = self.upscale(conv5_m)
        conv4_m_2 = self.conv4_1x1(conv4)
        conv4_m = conv4_m_1 + conv4_m_2
        conv4_p = self.conv4_3x3_1(conv4_m)
        conv4_p = self.conv4_3x3_2(conv4_p)

        conv3_m_1 = self.upscale(conv4_m)
        conv3_m_2 = self.conv3_1x1(conv3)
        conv3_m = conv3_m_1 + conv3_m_2
        conv3_p = self.conv3_3x3_1(conv3_m)
        conv3_p = self.conv3_3x3_2(conv3_p)

        conv2_m_1 = self.upscale(conv3_m)
        conv2_m_2 = self.conv2_1x1(conv2)
        conv2_m = conv2_m_1 + conv2_m_2
        conv2_p = self.conv2_3x3_1(conv2_m)
        conv2_p = self.conv2_3x3_2(conv2_p)

        return conv5_p, conv4_p, conv3_p, conv2_p


class FCN(tf.keras.Model):
    def __init__(self, n_classes=8, backbone=None, **kwargs):
        super().__init__(name="FCN", **kwargs)
        self.fpn = FPN(backbone)
        self.upscale_2x = tf.keras.layers.UpSampling2D()
        self.upscale_4x = tf.keras.layers.UpSampling2D((4, 4))
        self.upscale_8x = tf.keras.layers.UpSampling2D((8, 8))
        self.concat = tf.keras.layers.Concatenate()
        self.conv6 = tf.keras.layers.Conv2D(
            filters=(512), kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv7 = tf.keras.layers.Conv2D(
            filters=n_classes, kernel_size=(1, 1), padding="same", activation="relu"
        )
        self.upscale_final = tf.keras.layers.UpSampling2D(
            size=(4, 4), interpolation="bilinear"
        )

    def call(self, images, training=False):
        conv5_p, conv4_p, conv3_p, conv2_p = self.fpn(images, training=training)
        m_5 = self.upscale_8x(conv5_p)
        m_4 = self.upscale_4x(conv4_p)
        m_3 = self.upscale_2x(conv3_p)
        m_2 = conv2_p

        m_all = self.concat([m_2, m_3, m_4, m_5])
        m_all = self.conv6(m_all)
        m_all = self.conv7(m_all)
        m_all = self.upscale_final(m_all)

        return m_all


class FCN_ORIG(tf.keras.Model):
    def __init__(self, n_classes=8, backbone=None, **kwargs):
        super().__init__(name="FCN_ORIG", **kwargs)

        self.backbone = create_backbone_efficient()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=(n_classes), kernel_size=(1, 1), padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=(n_classes), kernel_size=(1, 1), padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=(n_classes), kernel_size=(1, 1), padding="same", activation="relu"
        )
        self.upscale2x_1 = tf.keras.layers.Convolution2DTranspose(
            filters=8,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            activation="relu",
        )
        self.upscale2x_2 = tf.keras.layers.Convolution2DTranspose(
            filters=8,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            activation="relu",
        )
        self.upscale2x_3 = tf.keras.layers.Convolution2DTranspose(
            filters=8,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            activation="relu",
        )
        self.upscale2x_4 = tf.keras.layers.Convolution2DTranspose(
            filters=8,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding="same",
            activation="relu",
        )

    def call(self, images, training=False):
        conv1_o, conv2_o, conv3_o, conv4_o = self.backbone(images, training=False)
        conv1_o = self.conv1(conv1_o)
        conv2_o = self.conv2(conv2_o)
        conv3_o = self.conv3(conv3_o)

        fcn_16x = self.upscale2x_1(conv4_o) + conv3_o
        fcn_8x = self.upscale2x_2(fcn_16x) + conv2_o
        fcn_4x = self.upscale2x_3(fcn_8x) + conv1_o
        final_output = self.upscale2x_4(fcn_4x)
        return final_output


class RescalingUnet(keras.layers.Layer):
    def __init__(self):
        super(RescalingUnet, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call(self, inputs):
        return ((inputs * (1 / 255.0)) - self.mean) / self.std


def combined_model_unetfpn(mode="multiply", n_classes=8):
    model_unet = sm.Unet(
        backbone_name="efficientnetb0",
        encoder_weights="imagenet",
        encoder_freeze=False,
        classes=n_classes,
        decoder_use_batchnorm=False,
    )
    model_fpn = FCN(n_classes)

    conv1x1 = keras.layers.Conv2D(n_classes, 1, padding="same", activation="softmax")
    input_layer = keras.layers.Input([None, None, 3])
    rescale_layer = RescalingUnet()

    if mode == "concat":
        concat = keras.layers.Concatenate()

    output_model_fcn = model_unet(rescale_layer(input_layer))
    output_model_fpn = model_fpn(input_layer)

    if mode == "multiply":
        output = output_model_fcn * output_model_fpn
    elif mode == "sum":
        output = output_model_fcn + output_model_fpn
    elif mode == "concat":
        output = concat([output_model_fcn, output_model_fpn])
    else:
        raise AssertionError("mode selected is not in the list")

    output_final = conv1x1(output)

    return keras.Model([input_layer], [output_final])


def combined_model_fcnfpn(mode="multi", n_classes=8):
    model_fcn = FCN_ORIG(n_classes)
    model_fpn = FCN(n_classes)

    conv1x1 = keras.layers.Conv2D(n_classes, 1, padding="same", activation="softmax")

    input_layer = keras.layers.Input([None, None, 3])
    output_model_fcn = model_fcn(input_layer)
    output_model_fpn = model_fpn(input_layer)
    output = output_model_fcn * output_model_fpn
    output_final = conv1x1(output)

    return keras.Model([input_layer], [output_final])


class NeuralnginConfig(AppConfig):
    name = "neuralngin"

    def ready(self):
        return
        # self.fcn = tf.keras.models.load_model(
        #     os.path.join(settings.BASE_DIR, "savedmodel/seagull_fpn")
        # )

    def predict(self, raw_io, model, infclass):
        """
        image: in str, decoded
        """

        image = tf.image.decode_image(raw_io, channels=3)
        image = tf.expand_dims(image, 0)
        image = tf.cast(image, tf.float32)
        print(image.shape, infclass)

        if model == "fpn":
            model = FCN(8)
            ckpt = tf.train.Checkpoint(model=model)
            ckptmg = tf.train.CheckpointManager(ckpt, f"trained_model/fpn", 5)
            ckptmg.restore_or_initialize()

            input_img_padded = tf.pad(image, [[0, 0], [8, 8], [0, 0], [0, 0]])
            output = model(input_img_padded, training=False)
            output = tf.math.softmax(output)
            output1 = tf.repeat(output[0, ..., infclass][..., tf.newaxis], 3, -1) * [
                30,
                128,
                128,
            ]

            image = tf.image.encode_jpeg(tf.cast(output1, tf.uint8))
            del model
            return b64encode(image.numpy())
        elif model == "unet":
            model = sm.Unet(
                backbone_name="efficientnetb0",
                encoder_weights="imagenet",
                encoder_freeze=False,
                activation="softmax",
                classes=8,
                decoder_use_batchnorm=False,
            )

            ckpt = tf.train.Checkpoint(model=model)
            ckptmg = tf.train.CheckpointManager(ckpt, f"trained_model/unet", 5)
            ckptmg.restore_or_initialize()

            input_img_padded = tf.pad(image, [[0, 0], [8, 8], [0, 0], [0, 0]])
            input_img_padded = sm.get_preprocessing("efficientnetb0")(input_img_padded)

            output = model(input_img_padded, training=True)
            output1 = tf.repeat(output[0, ..., infclass][..., tf.newaxis], 3, -1) * [
                30,
                128,
                128,
            ]

            image = tf.image.encode_jpeg(tf.cast(output1, tf.uint8))
            del model
            return b64encode(image.numpy())
        elif model == "fcn":
            model = FCN_ORIG(8)

            ckpt = tf.train.Checkpoint(model=model)
            ckptmg = tf.train.CheckpointManager(ckpt, f"trained_model/fcn8s", 5)
            ckptmg.restore_or_initialize()

            input_img_padded = tf.pad(image, [[0, 0], [8, 8], [0, 0], [0, 0]])
            output = model(input_img_padded, training=False)
            output = tf.math.softmax(output)
            output1 = tf.repeat(output[0, ..., infclass][..., tf.newaxis], 3, -1) * [
                30,
                128,
                128,
            ]

            image = tf.image.encode_jpeg(tf.cast(output1, tf.uint8))
            del model
            return b64encode(image.numpy())
        elif model == "ensem":
            model = combined_model_unetfpn()
            ckpt = tf.train.Checkpoint(model=model)
            ckptmg = tf.train.CheckpointManager(ckpt, f"trained_model/unetfpn", 5)
            ckptmg.restore_or_initialize()

            input_img_padded = tf.pad(image, [[0, 0], [8, 8], [0, 0], [0, 0]])
            output = model(input_img_padded, training=False)
            output1 = tf.repeat(output[0, ..., infclass][..., tf.newaxis], 3, -1) * [
                30,
                128,
                128,
            ]

            image = tf.image.encode_jpeg(tf.cast(output1, tf.uint8))
            del model
            return b64encode(image.numpy())
