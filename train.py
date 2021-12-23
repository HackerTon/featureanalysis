import argparse
import datetime
import glob
import os

import segmentation_models as sm
import tensorflow as tf

sm.set_framework("tf.keras")
sm.framework()
tf.random.set_seed(1024)


def get_seagull_path(istrain=True, directory=None):

    if istrain:
        trainimg = os.path.join(directory, "trainimg", "*.jpg")
        images = glob.glob(trainimg, recursive=True)
        trainmask = os.path.join(directory, "trainmask", "*.jpg")
        labels = glob.glob(trainmask, recursive=True)
    else:
        testimg = os.path.join(directory, "testimg", "*.jpg")
        images = glob.glob(testimg, recursive=True)
        testmask = os.path.join(directory, "testmask", "*.jpg")
        labels = glob.glob(testmask, recursive=True)

    print(len(images), len(labels))

    mask_set = set()
    image_set = set()
    for lbl in labels:
        lbl = lbl.split("/")[-1]
        mask_set.add(lbl)

    for img in images:
        img = img.split("/")[-1]
        image_set.add(img)

    complete_path = mask_set.intersection(image_set)
    print(
        f"IMG - LBL NUM: {len(image_set.difference(mask_set))}, Intersection: {len(complete_path)}"
    )

    return [i for i in complete_path]


def get_image_decode(image, label):
    image = tf.io.read_file(image, "image")
    label = tf.io.read_file(label, "label")

    image = tf.image.decode_image(image)
    label = tf.image.decode_image(label)

    return image, label


def get_mask(image, label):
    labels = []
    labels.append(label[:, :, 0] == 0)
    labels.append(label[:, :, 0] == 255)

    labels = tf.cast(labels, tf.float32)
    image = tf.cast(image, tf.float32)

    # must perform this
    return image, tf.transpose(labels, [1, 2, 0])


def create_ds(batch_size, ratio=0.8, directory=None):
    AUTOTUNE = tf.data.AUTOTUNE

    directory = os.path.join(directory)
    testimg_path = os.path.join(directory, "testimg") + "/"
    testmask_path = os.path.join(directory, "testmask") + "/"
    trainimg_path = os.path.join(directory, "trainimg") + "/"
    trainmask_path = os.path.join(directory, "trainmask") + "/"

    def path_2_test(path):
        return (
            testimg_path + path,
            testmask_path + path,
        )

    def path_2_train(path):
        return (
            trainimg_path + path,
            trainmask_path + path,
        )

    paths = get_seagull_path(directory=directory)
    ds1 = tf.data.Dataset.from_tensor_slices(paths)
    ds1 = ds1.map(path_2_train, AUTOTUNE)

    paths = get_seagull_path(False, directory=directory)
    ds2 = tf.data.Dataset.from_tensor_slices(paths)
    ds2 = ds2.map(path_2_test, AUTOTUNE)

    ds = ds1.concatenate(ds2)
    ds = ds.cache()

    takefortrain = int(23124 * ratio)
    trainds = ds.take(takefortrain)
    testds = ds.skip(takefortrain).take(23124 - takefortrain)

    trainds = trainds.shuffle(23124)

    trainds = trainds.map(get_image_decode, AUTOTUNE)
    trainds = trainds.map(get_mask, AUTOTUNE)
    testds = testds.map(get_image_decode, AUTOTUNE)
    testds = testds.map(get_mask, AUTOTUNE)

    # # batch and prefetch
    trainds = trainds.batch(batch_size)
    testds = testds.batch(batch_size)

    trainds = trainds.prefetch(AUTOTUNE)

    return trainds, testds


def create_backbone():
    _backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=[None, None, 3]
    )

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
    def __init__(self, **kwargs):
        super().__init__(name="Feature_Pyramid_Network", **kwargs)

        self.backbone = create_backbone()
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
        self.conv5_bn = tf.keras.layers.BatchNormalization()
        self.conv4_3x3_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv4_3x3_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv4_bn = tf.keras.layers.BatchNormalization()
        self.conv3_3x3_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv3_3x3_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv3_bn = tf.keras.layers.BatchNormalization()
        self.conv2_3x3_1 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv2_3x3_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv2_bn = tf.keras.layers.BatchNormalization()
        self.upscale = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, images, training=False):
        # 112x112, 56x56, 28x28, 14x14
        conv2, conv3, conv4, conv5 = self.backbone(images, training=training)
        conv5_m = self.conv5_1x1(conv5)
        conv5_p = self.conv5_3x3_1(conv5_m)
        conv5_p = self.conv5_3x3_2(conv5_p)
        conv5_p = self.conv5_bn(conv5_p, training=training)

        conv4_m_1 = self.upscale(conv5_m)
        conv4_m_2 = self.conv4_1x1(conv4)
        conv4_m = conv4_m_1 + conv4_m_2
        conv4_p = self.conv4_3x3_1(conv4_m)
        conv4_p = self.conv4_3x3_2(conv4_p)
        conv4_p = self.conv4_bn(conv4_p, training=training)

        conv3_m_1 = self.upscale(conv4_m)
        conv3_m_2 = self.conv3_1x1(conv3)
        conv3_m = conv3_m_1 + conv3_m_2
        conv3_p = self.conv3_3x3_1(conv3_m)
        conv3_p = self.conv3_3x3_2(conv3_p)
        conv3_p = self.conv3_bn(conv3_p, training=training)

        conv2_m_1 = self.upscale(conv3_m)
        conv2_m_2 = self.conv2_1x1(conv2)
        conv2_m = conv2_m_1 + conv2_m_2
        conv2_p = self.conv2_3x3_1(conv2_m)
        conv2_p = self.conv2_3x3_2(conv2_p)
        conv2_p = self.conv2_bn(conv2_p, training=training)
        return conv5_p, conv4_p, conv3_p, conv2_p


class FCN(tf.keras.Model):
    def __init__(self, n_classes=8, **kwargs):
        super().__init__(name="FCN", **kwargs)
        self.fpn = FPN()
        self.upscale_2x = tf.keras.layers.UpSampling2D()
        self.upscale_4x = tf.keras.layers.UpSampling2D((4, 4))
        self.upscale_8x = tf.keras.layers.UpSampling2D((8, 8))
        self.concat = tf.keras.layers.Concatenate()
        self.conv6 = tf.keras.layers.Conv2D(
            filters=(512), kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.bnorm = tf.keras.layers.BatchNormalization()
        self.conv7 = tf.keras.layers.Conv2D(
            filters=n_classes, kernel_size=(1, 1), padding="same", activation="relu"
        )
        self.upscale_final = tf.keras.layers.UpSampling2D(
            size=(4, 4), interpolation="bilinear"
        )
        self.final_activation = tf.keras.layers.Activation("softmax")

    def call(self, images, training=False):
        conv5_p, conv4_p, conv3_p, conv2_p = self.fpn(images, training=training)
        m_5 = self.upscale_8x(conv5_p)
        m_4 = self.upscale_4x(conv4_p)
        m_3 = self.upscale_2x(conv3_p)
        m_2 = conv2_p

        m_all = self.concat([m_2, m_3, m_4, m_5])
        m_all = self.conv6(m_all)
        m_all = self.bnorm(m_all, training=training)
        m_all = self.conv7(m_all)
        m_all = self.upscale_final(m_all)
        m_all = self.final_activation(m_all)

        return m_all


class FCN_ORIG(tf.keras.Model):
    def __init__(self, n_classes=8, **kwargs):
        super().__init__(name="FCN_ORIG", **kwargs)

        self.backbone = create_backbone()
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
            filters=n_classes,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            activation="relu",
        )
        self.upscale2x_2 = tf.keras.layers.Convolution2DTranspose(
            filters=n_classes,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            activation="relu",
        )
        self.upscale8x = tf.keras.layers.Convolution2DTranspose(
            filters=n_classes,
            kernel_size=(16, 16),
            strides=(8, 8),
            padding="same",
            activation="relu",
        )
        self.final_activation = tf.keras.layers.Activation("softmax")

    def call(self, images, training=False):
        _, conv1_o, conv2_o, conv3_o = self.backbone(images, training=training)
        conv1_o = self.conv1(conv1_o)
        conv2_o = self.conv2(conv2_o)
        conv3_o = self.conv3(conv3_o)

        fcn_16x = self.upscale2x_1(conv3_o) + conv2_o
        fcn_8x = self.upscale2x_2(fcn_16x) + conv1_o
        final_output = self.upscale8x(fcn_8x)
        final_output = self.final_activation(final_output)
        return final_output


# the network must OUTPUT in logits [-inf, inf]
# make sure input dimension is [B, H, W, C]
def Jindex(target, pred):
    intersection = tf.reduce_sum(target * pred, [0, 1, 2])
    union = tf.reduce_sum(target + pred, [0, 1, 2]) - intersection
    return tf.reduce_mean((intersection + 0.1) / (union + 0.1))


def Dice(target, pred):
    intersection = tf.reduce_sum(2 * pred * target, [0, 1, 2])
    union = tf.reduce_sum(pred + target, [0, 1, 2])
    return tf.reduce_mean((intersection + 0.1) / (union + 0.1))


def train(parsed):
    # this iteration is calculated fom 160 iteration from
    # paper
    n_epoch = 20
    n_classes = 2
    batch_size = parsed.batchsize
    trainds, testds = create_ds(batch_size, directory=parsed.path)

    if parsed.model == "fcn":
        print("Fcn selected!")
        model = FCN_ORIG(n_classes)
        prepro = lambda x: x
        lr_rate = 1e-4
    elif parsed.model == "unet":
        print("Unet selected!")
        model = sm.Unet(
            backbone_name="efficientnetb0",
            encoder_weights="imagenet",
            encoder_freeze=False,
            activation="softmax",
            classes=n_classes,
        )
        prepro = sm.get_preprocessing("efficientnetb0")
        lr_rate = 1e-6
    elif parsed.model == "fpn":
        print("Fpn selected!")
        model = FCN(n_classes)
        prepro = lambda x: x
        lr_rate = 1e-5

    # create tensorboard logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/tape/" + parsed.model + "/" + current_time + "/train"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_log_dir = "logs/tape/" + parsed.model + "/" + current_time + "/test"
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    optimizer = tf.keras.optimizers.Adam(lr_rate)
    focal_loss = sm.losses.CategoricalFocalLoss()
    dice_loss = sm.losses.DiceLoss(class_weights=[0.001, 0.999])

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckptmg = tf.train.CheckpointManager(
        ckpt, f"trained_model/seagull_{parsed.model}", 5
    )
    ckptmg.restore_or_initialize()

    # Real training
    train_iteration = 0
    iteration = 0
    sum_iou = 0
    sum_loss = 0
    ALPHA = 1.0

    for epoch in range(n_epoch):
        for bs_images, bs_labels in trainds:
            bs_images = prepro(bs_images)

            with tf.GradientTape() as t:
                output = model(bs_images)
                c_loss = dice_loss(bs_labels, output) + ALPHA * focal_loss(
                    bs_labels, output
                )

            grad = t.gradient(c_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            sum_loss = c_loss
            train_iteration += 1

            # calculate loss and IoU at iteration
            # this is train
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", c_loss, step=train_iteration)
                tf.summary.scalar(
                    "iou", sm.metrics.iou_score(bs_labels, output), step=train_iteration
                )

        for bs_images, bs_labels in testds:
            output = model(bs_images, training=False)
            sum_loss += (
                dice_loss(bs_labels, output) + ALPHA * focal_loss(bs_labels, output)
            ) * batch_size
            sum_iou += sm.metrics.iou_score(bs_labels, output) * batch_size
            iteration += batch_size

        # calculate validation loss and IoU
        # this is test
        with test_summary_writer.as_default():
            tf.summary.scalar("loss", sum_loss / iteration, step=train_iteration)
            tf.summary.scalar("iou", sum_iou / iteration, step=train_iteration)

        ckptmg.save()
        print(
            f"Epoch: {epoch + 1}, TEST_IOU: {sum_iou / iteration}, TEST_LOSS: {sum_loss / iteration}"
        )
        iteration = 0
        sum_iou = 0
        sum_loss = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["fcn", "unet", "fpn"])
    parser.add_argument("--batchsize", default=4, type=int)
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="path that contain of the dataset",
    )
    train(parser.parse_args())
