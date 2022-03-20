import os
import glob
from venv import create
import tensorflow as tf
import tensorflow.keras as keras


def get_seagull_path(istrain=True):
    directory = "/home/hackerton/Downloads/Airbus + Seagull Dataset/"

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


# added resizing
def get_image_decode(image, label):
    image = tf.io.read_file(image, "image")
    label = tf.io.read_file(label, "label")

    image = tf.image.decode_image(image)
    label = tf.image.decode_image(label)

    return image, label


def path_2_test(path):
    return (
        "/home/hackerton/Downloads/Airbus + Seagull Dataset/testimg/" + path,
        "/home/hackerton/Downloads/Airbus + Seagull Dataset/testmask/" + path,
    )


def path_2_train(path):
    return (
        "/home/hackerton/Downloads/Airbus + Seagull Dataset/trainimg/" + path,
        "/home/hackerton/Downloads/Airbus + Seagull Dataset/trainmask/" + path,
    )


def get_mask(image, label):
    labels = []
    labels.append(label[:, :, 0] == 0)
    labels.append(label[:, :, 0] == 255)

    labels = tf.cast(labels, tf.float32)
    image = tf.cast(image, tf.float32)

    # must perform this
    return image, tf.transpose(labels, [1, 2, 0])


def create_ds(batch_size, ratio=0.8):
    AUTOTUNE = tf.data.AUTOTUNE

    paths = get_seagull_path()
    ds1 = tf.data.Dataset.from_tensor_slices(paths)
    ds1 = ds1.map(path_2_train, AUTOTUNE)

    paths = get_seagull_path(False)
    ds2 = tf.data.Dataset.from_tensor_slices(paths)
    ds2 = ds2.map(path_2_test, AUTOTUNE)

    ds = ds1.concatenate(ds2)

    takefortrain = int(23124 * ratio)
    trainds = ds.take(takefortrain)
    testds = ds.skip(takefortrain).take(23124 - takefortrain)
    trainds = trainds.cache()
    testds = testds.cache()

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


def create_backbone_efficient():
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


def create_backbone_mobile():
    _backbone = keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False, input_shape=[None, None, 3]
    )

    outputs = [
        layer.output
        for layer in _backbone.layers
        if layer.name
        in [
            "block_1_depthwise_relu",
            "block_3_depthwise_relu",
            "block_12_depthwise_relu",
            "block_16_depthwise_relu",
        ]
    ]

    return tf.keras.Model(
        inputs=[_backbone.input], outputs=outputs, name="efficientb0_backbone"
    )


class FPN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(name="Feature_Pyramid_Network", **kwargs)

        self.backbone = create_backbone_mobile()
        initializer = keras.initializers.he_normal()

        self.conv5_1x1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(1, 1),
            padding="same",
            kernel_regularizer=keras.initializers.he_normal(),
        )
        self.conv4_1x1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(1, 1),
            padding="same",
            kernel_regularizer=initializer,
        )
        self.conv3_1x1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(1, 1),
            padding="same",
            kernel_regularizer=initializer,
        )
        self.conv2_1x1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(1, 1),
            padding="same",
            kernel_regularizer=initializer,
        )
        self.conv5_3x3_1 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=initializer,
        )
        self.conv5_3x3_2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=initializer,
        )
        self.conv5_bn = tf.keras.layers.BatchNormalization()
        self.conv4_3x3_1 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=initializer,
        )
        self.conv4_3x3_2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=initializer,
        )
        self.conv4_bn = tf.keras.layers.BatchNormalization()
        self.conv3_3x3_1 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=initializer,
        )
        self.conv3_3x3_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv3_bn = tf.keras.layers.BatchNormalization()
        self.conv2_3x3_1 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=initializer,
        )
        self.conv2_3x3_2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=initializer,
        )
        self.conv2_bn = tf.keras.layers.BatchNormalization()
        self.upscale = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, images, training=False):
        # 112x112, 56x56, 28x28, 14x14
        conv2, conv3, conv4, conv5 = self.backbone(images, training=False)
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
        initializer = keras.initializers.he_normal()
        self.upscale_2x = tf.keras.layers.UpSampling2D()
        self.upscale_4x = tf.keras.layers.UpSampling2D((4, 4))
        self.upscale_8x = tf.keras.layers.UpSampling2D((8, 8))
        self.concat = tf.keras.layers.Concatenate()
        self.conv6 = tf.keras.layers.Conv2D(
            filters=(512),
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=initializer,
        )
        self.bnorm = tf.keras.layers.BatchNormalization()
        self.conv7 = tf.keras.layers.Conv2D(
            filters=n_classes,
            kernel_size=(1, 1),
            padding="same",
            activation="relu",
            kernel_regularizer=initializer,
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
        self.upscale8x = tf.keras.layers.Convolution2DTranspose(
            filters=8,
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


def Test():
    print("this is new")
