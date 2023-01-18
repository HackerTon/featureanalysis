import datetime
import time

# import segmentation_models as sm
import tensorflow as tf

# import tensorflow.keras as keras
from tensorflow import keras

from model import RescalingUnet, SingleModel
from preprocessing import UavidDataset
from metrics import jindex_class

# Set global seed for reproducibility
tf.random.set_seed(1024)

# Set segmentation_models to use TF framework
# sm.set_framework("tf.keras")


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


@tf.function
def evaluate(model, dataset):
    class_avg_iou = 0.0
    iteration = 0.0
    for bs_images, bs_labels in dataset:
        output = model(bs_images, training=False)
        output = tf.nn.softmax(output)
        class_avg_iou += jindex_class(bs_labels, output)
        iteration += 1
    return class_avg_iou, iteration


def mainSingle():
    n_class = 8
    batch_size = 8
    trainds, testds = UavidDataset.create_ds(batch_size=batch_size)
    model = SingleModel.FPN(n_class=n_class)
    model_name = model.name

    # Load latest checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    ckptmg = tf.train.CheckpointManager(
        ckpt,
        f"trained_model/{model_name}",
        max_to_keep=None,
    )

    if ckptmg.latest_checkpoint is not None:
        ckpt.restore(ckptmg.latest_checkpoint).expect_partial()
        print("Checkpoint loaded!")

    # Evaluating on train dataset
    initial_time = time.time()
    class_avg_iou, iteration = evaluate(model, trainds)
    time_taken_second = time.time() - initial_time
    print(f"Mean training IoU : {tf.reduce_mean(class_avg_iou / iteration)}")
    print(f"Class training IoU : {class_avg_iou / iteration}")
    print(f"Time taken : {time_taken_second}s")
    print()

    # Evaluating on test dataset
    initial_time = time.time()
    class_avg_iou, iteration = evaluate(model, testds)
    time_taken_second = time.time() - initial_time
    print(f"Mean testing IoU : {tf.reduce_mean(class_avg_iou / iteration)}")
    print(f"Class testing IoU : {class_avg_iou / iteration}")
    print(f"Time taken : {time_taken_second}s")


if __name__ == "__main__":
    mainSingle()
