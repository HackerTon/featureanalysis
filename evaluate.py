import datetime
import time

import segmentation_models as sm
import tensorflow as tf
import tensorflow.keras as keras

from model import RescalingUnet, SingleModel
from preprocessing import UavidDataset

# Set global seed for reproducibility
tf.random.set_seed(1024)

# Set segmentation_models to use TF framework
sm.set_framework("tf.keras")


def combined_model(mode="multi", n_class=8):
    model_unet = sm.Unet(
        backbone_name="efficientnetb0",
        encoder_weights="imagenet",
        encoder_freeze=False,
        classes=n_class,
        decoder_use_batchnorm=False,
    )
    model_fpn = SingleModel.FPN(n_class)
    conv1x1 = keras.layers.Conv2D(
        n_class,
        1,
        padding="same",
        activation="softmax",
    )
    rescale_layer = RescalingUnet()
    input_layer = keras.layers.Input([None, None, 3])

    output_model_fcn = model_unet(rescale_layer(input_layer))
    output_model_fpn = model_fpn(input_layer)
    output = (output_model_fcn + output_model_fpn) / 2
    output_final = conv1x1(output)

    return keras.Model([input_layer], [output_final], name="FPN_UNET_MEAN")


def mainSingle():
    n_class = 8
    batch_size = 1
    trainds, testds = UavidDataset.create_ds(batch_size=batch_size)
    model = SingleModel.FPN(n_class=n_class)
    model_name = model.name

    ckpt = tf.train.Checkpoint(model=model)
    ckptmg = tf.train.CheckpointManager(ckpt, f"trained_model_test/{model_name}", 5)
    ckptmg.restore_or_initialize()

    avg_iou = 0
    iteration = 0
    for bs_images, bs_labels in trainds:
        output = model(bs_images, training=False)
        output = tf.nn.softmax(output)
        avg_iou += sm.metrics.iou_score(bs_labels, output)
        iteration += 1

    print(f"Training IoU : {avg_iou / iteration}")
    avg_iou = 0
    iteration = 0

    for bs_images, bs_labels in testds:
        output = model(bs_images, training=False)
        output = tf.nn.softmax(output)
        avg_iou += sm.metrics.iou_score(bs_labels, output)
        iteration += 1


if __name__ == "__main__":
    # mainCombined()
    mainSingle()
