import datetime

import tensorflow as tf
from tensorflow.python.keras import backend as K

from metrics import dice_loss, jindex_class
from model import MultiModel, SingleModel
from preprocessing import UavidDataset, UavidDatasetOld


@tf.function
def backprop(
    model,
    bs_images,
    bs_labels,
):
    with tf.GradientTape() as tape:
        output = model(bs_images, training=True)
        output = tf.nn.softmax(output)
        c_loss = dice_loss(bs_labels, output)

    grad = tape.gradient(c_loss, model.trainable_variables)
    iou = tf.reduce_mean(tf.reduce_mean(jindex_class(bs_labels, output), axis=-1))
    average_c_loss = tf.reduce_mean(c_loss)

    return average_c_loss, iou, grad


@tf.function()
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


def trainUniversal(model_choice=0, batch_size=8, test_batch_size=12):
    # Training parameter
    n_epoch = 20
    n_class = 8

    # Setting seed for reproducibility
    tf.random.set_seed(1024)

    trainds, testds = UavidDatasetOld.create_ds(
        batch_size=batch_size,
        test_batch_size=test_batch_size,
    )

    if model_choice == 0:
        model = SingleModel.FCN(n_class)
    elif model_choice == 1:
        model = SingleModel.UNET(n_class)
    elif model_choice == 2:
        model = SingleModel.FPN(n_class)
    elif model_choice == 3:
        model = MultiModel.FpnUnetProduct(n_class)
    elif model_choice == 4:
        model = MultiModel.FpnUnetSummation(n_class)
    elif model_choice == 5:
        model = MultiModel.FpnUnetConcatenation(n_class)
    elif model_choice == 6:
        model = MultiModel.FpnFcnConcatenation(n_class)
    else:
        assert "No model chosen"

    # Initial the model with size
    model(tf.random.uniform([1, 512, 512, 3]))

    model_name = model.name
    optimizer = tf.keras.optimizers.Adam(5e-5)

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
        initial_time = tf.timestamp()
        for bs_images, bs_labels in trainds:
            c_loss, iou, grad = backprop(
                model,
                bs_images,
                bs_labels,
            )

            print(c_loss, iou)

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
            testing_loss += current_loss
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
    for i in range(2, 8):
        # Change batch_size to 8 and 16 for single network
        if i < 3:
            trainUniversal(model_choice=i)
        else:
        # Change batch_size to 8 and 16 for single network
            trainUniversal(model_choice=i, batch_size=4, test_batch_size=6)
