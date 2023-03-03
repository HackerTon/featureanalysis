import datetime

import tensorflow as tf
from tensorflow import keras

from metrics import dice_coef, dice_loss, jindex_class
from model import RescalingUnet, SingleModel
from preprocessing import UavidDataset

# Set global seed for reproducibility
tf.random.set_seed(1024)


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


# def mainCombined():
#     # this iteration is calculated fom 160 iteration from
#     # paper
#     n_epoch = 20
#     n_class = 8
#     batch_size = 2
#     trainds, testds = UavidDataset.create_ds(batch_size=batch_size)
#     model = combined_model()
#     model_name = model.name

#     optimizer = keras.optimizers.Adam(1e-5)
#     focal_loss = sm.losses.CategoricalFocalLoss()
#     dice_loss = sm.losses.DiceLoss()

#     ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
#     ckptmg = tf.train.CheckpointManager(ckpt, f"trained_model_test/{model_name}", 5)
#     ckptmg.restore_or_initialize()

#     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#     train_log_dir = f"log_test/{model_name}/{current_time}/train"
#     train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#     test_log_dir = f"log_test/{model_name}/{current_time}/test"
#     test_summary_writer = tf.summary.create_file_writer(test_log_dir)

#     # Real training
#     train_iteration = 0
#     iteration = 0
#     ALPHA = 1.0

#     for epoch in range(n_epoch):
#         for bs_images, bs_labels in trainds:
#             initial_time = time.time()
#             with tf.GradientTape() as t:
#                 output = model(bs_images, training=True)
#                 c_loss = dice_loss(bs_labels, output) + ALPHA * focal_loss(
#                     bs_labels, output
#                 )

#             grad = t.gradient(c_loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grad, model.trainable_variables))
#             sum_loss = c_loss
#             train_iteration += 1
#             final_time = time.time()

#             # calculate loss and IoU at iteration
#             # this is train
#             with train_summary_writer.as_default():
#                 tf.summary.scalar(
#                     "loss",
#                     c_loss,
#                     step=train_iteration,
#                 )
#                 tf.summary.scalar(
#                     "iou",
#                     sm.metrics.iou_score(bs_labels, output),
#                     step=train_iteration,
#                 )
#                 tf.summary.scalar(
#                     "timer per step",
#                     (final_time - initial_time) / batch_size,
#                     step=train_iteration,
#                 )

#         # for bs_images, bs_labels in testds:
#         #     output = model(bs_images, training=False)
#         #     sum_loss += (
#         #         dice_loss(bs_labels, output) + ALPHA * focal_loss(bs_labels, output)
#         #     ) * batch_size
#         #     sum_iou += sm.metrics.iou_score(bs_labels, output) * batch_size
#         #     iteration += batch_size

#         # # calculate validation loss and IoU
#         # # this is test
#         # with test_summary_writer.as_default():
#         #     tf.summary.scalar("loss", sum_loss / iteration, step=train_iteration)
#         #     tf.summary.scalar("iou", sum_iou / iteration, step=train_iteration)

#         iteration = 0
#         sum_iou = 0
#         sum_loss = 0
#         ckptmg.save()


@tf.function
def backprop(
    model,
    optimizer,
    bs_images,
    bs_labels,
):
    with tf.GradientTape() as tape:
        output = model(bs_images, training=True)
        output = tf.nn.softmax(output)
        c_loss = dice_loss(bs_labels, output)

    grad = tape.gradient(c_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    iou = tf.reduce_mean(jindex_class(bs_labels, output))
    average_c_loss = tf.reduce_mean(c_loss)

    return average_c_loss, iou


@tf.function
def evaluate(
    model,
    bs_images,
    bs_labels,
):
    output = model(bs_images, training=False)
    output = tf.nn.softmax(output)
    testing_loss = dice_loss(bs_labels, output)
    iou = tf.reduce_mean(jindex_class(bs_labels, output))
    return testing_loss, iou


def mainSingle():
    n_epoch = 20
    n_class = 8
    batch_size = 1
    test_batch_size = 4
    trainds, testds = UavidDataset.create_ds(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
    )
    model = SingleModel.FPN(n_class=n_class)
    model_name = model.name

    optimizer = keras.optimizers.Adam(0.00001)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckptmg = tf.train.CheckpointManager(
        ckpt,
        f"trained_model/{model_name}",
        max_to_keep=None,
    )
    if ckptmg.latest_checkpoint is not None:
        ckpt.restore(ckptmg.latest_checkpoint).expect_partial()
        print("Checkpoint loaded!")

    current_time = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")

    train_log_dir = f"log_test/{model_name}/{current_time}/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_log_dir = f"log_test/{model_name}/{current_time}/test"
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    train_iteration = 0

    iteration = 0
    for epoch in range(n_epoch):
        # if epoch == 0:
        #     model.freeze_backbone()
        # elif epoch == 1:
        #     model.unfreeze_backbone()
        #     optimizer.lr.assign(0.00001)

        initial_time = tf.timestamp()
        for bs_images, bs_labels in trainds:
            c_loss, iou = backprop(
                model,
                optimizer,
                bs_images,
                bs_labels,
            )
            train_iteration += 1
            iteration += 1
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    "Loss",
                    c_loss,
                    step=train_iteration,
                )
                tf.summary.scalar(
                    "IoU",
                    iou,
                    step=train_iteration,
                )
        final_time = tf.timestamp()
        with train_summary_writer.as_default():
            tf.summary.scalar(
                "Rate of fnbprob",
                (batch_size * iteration) / (final_time - initial_time),
                step=train_iteration,
            )

        iteration = 0
        testing_loss = 0
        iou_score = 0
        initial_time = tf.timestamp()
        for bs_images, bs_labels in testds:
            current_loss, current_iou = evaluate(model, bs_images, bs_labels)
            testing_loss += current_loss
            iou_score += current_iou
            iteration += 1
        final_time = tf.timestamp()

        with test_summary_writer.as_default():
            tf.summary.scalar("Loss", testing_loss / iteration, step=train_iteration)
            tf.summary.scalar("IoU", iou_score / iteration, step=train_iteration)
            tf.summary.scalar(
                "Rate of fnbprob",
                (test_batch_size * iteration) / (final_time - initial_time),
                step=train_iteration,
            )
        ckptmg.save()


if __name__ == "__main__":
    mainSingle()
