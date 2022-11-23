import tensorflow as tf
import tensorflow.keras as keras


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
        inputs=[_backbone.input],
        outputs=outputs,
        name="efficientb0_backbone",
    )


class SingleModel:
    class FPN(tf.keras.layers.Layer):
        def __init__(self, n_class=8, *args, **kwargs):
            super().__init__(name="Feature_Pyramid_Network", *args, **kwargs)
            self.backbone = create_backbone_efficient()
            self.conv5_1x1 = tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(1, 1),
                padding="same",
            )
            self.conv4_1x1 = tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(1, 1),
                padding="same",
            )
            self.conv3_1x1 = tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(1, 1),
                padding="same",
            )
            self.conv2_1x1 = tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(1, 1),
                padding="same",
            )
            self.conv5_3x3_1 = tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
            self.conv5_3x3_2 = tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
            self.conv4_3x3_1 = tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
            self.conv4_3x3_2 = tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
            self.conv3_3x3_1 = tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
            self.conv3_3x3_2 = tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
            self.conv2_3x3_1 = tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
            self.conv2_3x3_2 = tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
            self.upscale2x = keras.layers.UpSampling2D()
            self.upscale4x = keras.layers.UpSampling2D((4, 4))
            self.upscale8x = keras.layers.UpSampling2D((8, 8))
            self.concatenate = tf.keras.layers.Concatenate()
            self.conv6 = tf.keras.layers.Conv2D(
                filters=(512), kernel_size=(3, 3), padding="same", activation="relu"
            )
            self.conv7 = tf.keras.layers.Conv2D(
                filters=n_class, kernel_size=(1, 1), padding="same", activation="relu"
            )
            self.upscalefinal = tf.keras.layers.UpSampling2D(
                size=(4, 4), interpolation="bilinear"
            )

        def call(self, images, training=False):
            # 112x112, 56x56, 28x28, 14x14
            conv2, conv3, conv4, conv5 = self.backbone(images, training=training)
            conv5_m = self.conv5_1x1(conv5)
            conv5_p = self.conv5_3x3_1(conv5_m)
            conv5_p = self.conv5_3x3_2(conv5_p)

            conv4_m_1 = self.upscale2x(conv5_m)
            conv4_m_2 = self.conv4_1x1(conv4)
            conv4_m = conv4_m_1 + conv4_m_2
            conv4_p = self.conv4_3x3_1(conv4_m)
            conv4_p = self.conv4_3x3_2(conv4_p)

            conv3_m_1 = self.upscale2x(conv4_m)
            conv3_m_2 = self.conv3_1x1(conv3)
            conv3_m = conv3_m_1 + conv3_m_2
            conv3_p = self.conv3_3x3_1(conv3_m)
            conv3_p = self.conv3_3x3_2(conv3_p)

            conv2_m_1 = self.upscale2x(conv3_m)
            conv2_m_2 = self.conv2_1x1(conv2)
            conv2_m = conv2_m_1 + conv2_m_2
            conv2_p = self.conv2_3x3_1(conv2_m)
            conv2_p = self.conv2_3x3_2(conv2_p)

            # Middle part is below
            m_5 = self.upscale8x(conv5_p)
            m_4 = self.upscale4x(conv4_p)
            m_3 = self.upscale2x(conv3_p)
            m_2 = conv2_p

            m_all = self.concatenate([m_2, m_3, m_4, m_5])
            m_all = self.conv6(m_all)
            m_all = self.conv7(m_all)
            m_all = self.upscalefinal(m_all)

            return m_all


class RescalingUnet(keras.layers.Layer):
    def __init__(self):
        super(RescalingUnet, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call(self, inputs):
        return ((inputs * (1 / 255.0)) - self.mean) / self.std
