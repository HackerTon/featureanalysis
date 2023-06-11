import datetime

import tensorflow as tf
from tensorflow.python.keras import backend as K
from segmentation_models.losses import categorical_focal_dice_loss
from segmentation_models.metrics import iou_score

from metrics import jindex_class, dice_loss
from model import MultiModel, SingleModel
from preprocessing import UavidDataset

# Remark
# Using tf.function actually makes your computation in graph mode
# Problem that was facing, cannot change learning rate
# Training not converging

@tf.function
def backprop(
    model,
    bs_images,
    bs_labels,
):
    with tf.GradientTape() as tape:
        output = model(bs_images, training=True)
        softmaxed_output = tf.nn.softmax(output)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                y_true=bs_labels,
                y_pred=softmaxed_output,
            ),
            axis=[1, 2],
        ) + dice_loss(bs_labels, softmaxed_output)
        loss = categorical_focal_dice_loss(bs_labels, softmaxed_output)
    grad = tape.gradient(loss, model.trainable_variables)
    iou = tf.reduce_mean(jindex_class(bs_labels, softmaxed_output))
    return tf.reduce_mean(loss), iou, grad
    # return loss, iou, grad


@tf.function()
def evaluate(
    model,
    bs_images,
    bs_labels,
):
    output = model(bs_images, training=False)
    softmaxed_output = tf.nn.softmax(output)
    loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            y_true=bs_labels,
            y_pred=softmaxed_output,
        ),
        axis=[1, 2],
    ) + dice_loss(bs_labels, softmaxed_output)
    # loss = categorical_focal_dice_loss(bs_labels, softmaxed_output)
    iou = tf.reduce_mean(jindex_class(bs_labels, softmaxed_output))
    return tf.reduce_mean(loss), iou
    # return loss, iou

def trainUniversal(model_choice=0, batch_size=8, test_batch_size=16):
    # Training parameter
    n_epoch = 80
    n_class = 8

    # Setting seed for reproducibility
    tf.random.set_seed(24)

    trainds, testds = UavidDataset.create_ds(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        # path_dir='/pool/storage/uavid_v1.5_official_release_image/'
    )

    if model_choice == 0:
        model = SingleModel.FCN(n_class)
        # model.freeze_backbone()
    elif model_choice == 1:
        model = SingleModel.UNET(n_class)
    elif model_choice == 2:
        model = SingleModel.FPN(n_class)
        # model.freeze_backbone()
    elif model_choice == 3:
        model = MultiModel.FpnUnetProduct(n_class)
    elif model_choice == 4:
        model = MultiModel.FpnUnetSummation(n_class)
    elif model_choice == 5:
        model = MultiModel.FpnUnetConcatenation(n_class)
    elif model_choice == 6:
        model = MultiModel.FpnFcnConcatenation(n_class)
        # model.freeze_backbone()
    else:
        assert "No model chosen"

    model_name = model.name
    # Learning rate for FCN is set to 0.0001
    # While learning rate for others are set to default 0.001
    optimizer = tf.keras.optimizers.Adam(0.00001)

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
        # if epoch == 1:
            # model.unfreeze_backbone()
            # optimizer.learning_rate.assign(0.00001)

        initial_time = tf.timestamp()
        for bs_images, bs_labels in trainds:
            c_loss, iou, grad = backprop(
                model,
                bs_images,
                bs_labels,
            )
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

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
                "Number of forward-pass per second",
                (batch_size * iteration) / (final_time - initial_time),
                step=train_iteration,
            )

        iteration = 0
        testing_loss = 0
        iou_score = 0
        initial_time = tf.timestamp()
        for bs_images, bs_labels in testds:
            current_loss, current_iou = evaluate(model, bs_images, bs_labels)
            testing_loss += tf.reduce_mean(current_loss)
            iou_score += current_iou
            iteration += 1
        final_time = tf.timestamp()

        with test_summary_writer.as_default():
            tf.summary.scalar(
                "Loss", tf.reduce_mean(testing_loss) / iteration, step=train_iteration
            )
            tf.summary.scalar("IoU", iou_score / iteration, step=train_iteration)
            tf.summary.scalar(
                "Number of forward-pass per second",
                (test_batch_size * iteration) / (final_time - initial_time),
                step=train_iteration,
            )
        ckptmg.save()

    # Clear session for this function
    K.clear_session()
    del model


if __name__ == "__main__":
    for i in range(2, 3):
        # Change batch_size to 8 and 16 for single network
        if i < 3:
            trainUniversal(model_choice=i, batch_size=1, test_batch_size=4)
            # trainUniversal(model_choice=i)
        else:
            # Change batch_size to 8 and 16 for single network
            trainUniversal(model_choice=i, batch_size=4, test_batch_size=6)
