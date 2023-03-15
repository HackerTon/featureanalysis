import time
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras import backend as K

from metrics import dice_coef, jindex_class
from model import MultiModel, SingleModel
from preprocessing import UavidDataset



def pretty_print(data_numpy):
    label = UavidDataset.labels()
    for i in range(data_numpy.shape[0]):
        print(f'{label[i]} ->  {data_numpy[i]:.4f}')
        

@tf.function()
def evaluate(
    model,
    bs_images,
    bs_labels,
):
    output = model(bs_images, training=False)
    output = tf.nn.softmax(output)
    iou_index = tf.reduce_mean(jindex_class(bs_labels, output), axis=0)
    dice_index = tf.reduce_mean(dice_coef(bs_labels, output), axis=0)
    return iou_index, dice_index


@tf.function()
def evaluateV2(
    model,
    dataset,
):
    sum_iou = tf.zeros([8])
    sum_dice = tf.zeros([8])
    iteration = tf.zeros([1])
    for bs_images, bs_labels in dataset:
        output = model(bs_images, training=False)
        output = tf.nn.softmax(output)
        sum_iou += tf.reduce_mean(jindex_class(bs_labels, output), axis=0)
        sum_dice += tf.reduce_mean(dice_coef(bs_labels, output), axis=0)
        iteration += 1
    return sum_iou, sum_dice, iteration


def evaluation(model_choice=0, batch_size=8, test_batch_size=16, results_df=None):
    # Parameters
    n_class = 8

    # Setting seed for reproducibility
    tf.random.set_seed(1024)

    trainds, testds = UavidDataset.create_ds(
        batch_size=test_batch_size,
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

    ckpt = tf.train.Checkpoint(model=model)
    ckptmg = tf.train.CheckpointManager(
        ckpt,
        f"trained_model/{model_name}",
        max_to_keep=None,
    )

    if ckptmg.latest_checkpoint is not None:
        ckpt.restore(ckptmg.latest_checkpoint).expect_partial()
        print("Checkpoint loaded!")
    else:
        print(f'Training checkpoint for {model_name} not found.')

    # Train dataset phase
    iteration = 0
    sum_iou_training = np.zeros([n_class])
    sum_dice_training = np.zeros([n_class])

    initial_time = time.time()
    sum_iou, sum_dice, iteration = evaluateV2(model, trainds)
    sum_iou_training += sum_iou
    sum_dice_training += sum_dice

    # for bs_images, bs_labels in trainds.take(1):
    #     iou_index, dice_index = evaluate(model, bs_images, bs_labels)
    #     sum_iou_training += iou_index.numpy()
    #     sum_dice_training += dice_index.numpy()
    #     iteration += 1
    time_taken_second = time.time() - initial_time
    current_result = [model_name]
    current_result += [x for x in (sum_iou_training / iteration).numpy()]
    current_result += [x for x in (sum_dice_training/ iteration).numpy()]

    print(f'Model name : {model_name}')
    print(f'Batch size : {test_batch_size}')
    print(f"Time taken : {time_taken_second}s")
    print(f"Mean training IoU : {np.mean(sum_iou_training / iteration):.4f}")
    print('Class training IoU : ')
    pretty_print(sum_iou_training /iteration)
    print(f"Mean training Dice : {np.mean(sum_dice_training / iteration):.4f}")
    print('Class training Dice : ')
    pretty_print(sum_dice_training /iteration)
    print()

    # Test dataset phase
    iteration = 0
    sum_iou_testing = np.zeros([n_class])
    sum_dice_testing = np.zeros([n_class])

    initial_time = time.time()
    sum_iou, sum_dice, iteration = evaluateV2(model, testds)
    sum_iou_testing += sum_iou
    sum_dice_testing += sum_dice

    # for bs_images, bs_labels in testds.take(1):
    #     iou_index, dice_index = evaluate(model, bs_images, bs_labels)
    #     sum_iou_testing += iou_index.numpy()
    #     sum_dice_testing += dice_index.numpy()
    #     iteration += 1
    time_taken_second = time.time() - initial_time

    current_result += [x for x in (sum_iou_testing / iteration).numpy()]
    current_result += [x for x in (sum_dice_testing/ iteration).numpy()]

    print(f"Time taken : {time_taken_second}s")
    print(f"Mean testing IoU : {np.mean(sum_iou_testing / iteration):.4f}")
    print('Class testing IoU : ')
    pretty_print(sum_iou_testing /iteration)
    print(f"Mean testing Dice : {np.mean(sum_dice_testing / iteration):.4f}")
    print('Class testing Dice : ')
    pretty_print(sum_dice_testing /iteration)
    print()

    # Clear session for this function
    K.clear_session()
    del model

    # Combine current results and return it
    return pd.concat([results_df, pd.DataFrame([current_result], columns=results_df.columns)])


if __name__ == "__main__":
    label = ['name'] 
    label += [f'train_iou_{x}' for x in UavidDataset.labels()]
    label += [f'train_dice_{x}' for x in UavidDataset.labels()]
    label += [f'test_iou_{x}' for x in UavidDataset.labels()]
    label += [f'test_dice_{x}' for x in UavidDataset.labels()]
    results_df = pd.DataFrame(columns=label)

    for i in range(7):
        results_df = evaluation(model_choice=i, results_df=results_df)

    results_df.to_csv('all_results.csv')

