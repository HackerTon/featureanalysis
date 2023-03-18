import segmentation_models as sm
import tensorflow as tf


def create_backbone_efficient():
    _backbone = tf.keras.applications.EfficientNetB0(include_top=False)
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
    class FPN(tf.keras.Model):
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
            self.upscale2x = tf.keras.layers.UpSampling2D((2, 2))
            self.upscale4x = tf.keras.layers.UpSampling2D((4, 4))
            self.upscale8x = tf.keras.layers.UpSampling2D((8, 8))
            self.concatenate = tf.keras.layers.Concatenate()
            self.conv6 = tf.keras.layers.Conv2D(
                filters=(512),
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
            self.conv7 = tf.keras.layers.Conv2D(
                filters=n_class,
                kernel_size=(1, 1),
                padding="same",
                activation="relu",
            )
            self.upscalefinal = tf.keras.layers.UpSampling2D(
                size=(4, 4),
                interpolation="bilinear",
            )
            self.add = tf.keras.layers.Add()

        def freeze_backbone(self):
            for layer in self.backbone.layers:
                layer.trainable = False

        def unfreeze_backbone(self):
            for layer in self.backbone.layers:
                layer.trainable = True

        def call(self, images, training=False):
            # Input is 448x448
            # 112x112, 56x56, 28x28, 14x14
            # Training argument for backbone passed must be set to False.
            # This is because we do not want to update the batch normalization
            # Layer of EfficientNetB0
            conv2, conv3, conv4, conv5 = self.backbone(images, training=False)
            conv5_m = self.conv5_1x1(conv5)
            conv5_p = self.conv5_3x3_1(conv5_m)
            conv5_p = self.conv5_3x3_2(conv5_p)

            conv4_m_1 = self.upscale2x(conv5_m)
            conv4_m_2 = self.conv4_1x1(conv4)
            conv4_m = self.add([conv4_m_1, conv4_m_2])
            conv4_p = self.conv4_3x3_1(conv4_m)
            conv4_p = self.conv4_3x3_2(conv4_p)

            conv3_m_1 = self.upscale2x(conv4_m)
            conv3_m_2 = self.conv3_1x1(conv3)
            conv3_m = self.add([conv3_m_1, conv3_m_2])
            conv3_p = self.conv3_3x3_1(conv3_m)
            conv3_p = self.conv3_3x3_2(conv3_p)

            conv2_m_1 = self.upscale2x(conv3_m)
            conv2_m_2 = self.conv2_1x1(conv2)
            conv2_m = self.add([conv2_m_1, conv2_m_2])
            conv2_p = self.conv2_3x3_1(conv2_m)
            conv2_p = self.conv2_3x3_2(conv2_p)

            m_5 = self.upscale8x(conv5_p)
            m_4 = self.upscale4x(conv4_p)
            m_3 = self.upscale2x(conv3_p)
            m_2 = conv2_p

            m_all = self.concatenate([m_2, m_3, m_4, m_5])
            m_all = self.conv6(m_all)
            m_all = self.conv7(m_all)
            m_all = self.upscalefinal(m_all)

            return m_all

    class UNET(tf.keras.Model):
        def __init__(self, n_classes=8, backbone=None, **kwargs):
            super().__init__(name="UNET", **kwargs)
            self.model = sm.Unet(
                backbone_name="efficientnetb0",
                encoder_weights="imagenet",
                encoder_freeze=False,
                activation="linear",
                classes=n_classes,
                decoder_use_batchnorm=False,
            )
            self.preprocessing = RescalingUnet()

        def call(self, images, training=False):
            ppreprocess_input = self.preprocessing(images)
            return self.model(ppreprocess_input, training)

    class FCN(tf.keras.Model):
        def __init__(self, n_classes=8, backbone=None, **kwargs):
            super().__init__(name="Fully_Convolutional_Network", **kwargs)

            self.backbone = create_backbone_efficient()
            self.conv1 = tf.keras.layers.Conv2D(
                filters=(n_classes),
                kernel_size=(1, 1),
                padding="same",
                activation="relu",
            )
            self.conv2 = tf.keras.layers.Conv2D(
                filters=(n_classes),
                kernel_size=(1, 1),
                padding="same",
                activation="relu",
            )
            self.conv3 = tf.keras.layers.Conv2D(
                filters=(n_classes),
                kernel_size=(1, 1),
                padding="same",
                activation="relu",
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
            self.add = tf.keras.layers.Add()

        def freeze_backbone(self):
            for layer in self.backbone.layers:
                layer.trainable = False

        def unfreeze_backbone(self):
            for layer in self.backbone.layers:
                layer.trainable = True

        def call(self, images, training=False):
            conv1_o, conv2_o, conv3_o, conv4_o = self.backbone(
                images, training=False
            )
            conv1_o = self.conv1(conv1_o)
            conv2_o = self.conv2(conv2_o)
            conv3_o = self.conv3(conv3_o)

            fcn_16x = self.add([self.upscale2x_1(conv4_o), conv3_o])
            fcn_8x = self.add([self.upscale2x_2(fcn_16x), conv2_o])
            fcn_4x = self.add([self.upscale2x_3(fcn_8x), conv1_o])
            final_output = self.upscale2x_4(fcn_4x)
            return final_output


class MultiModel:
    # FPN + UNET
    class FpnUnetProduct(tf.keras.Model):
        def __init__(self, n_class=8):
            super().__init__(name="Fpn_Unet_Product")

            self.fpn = SingleModel.FPN(n_class)
            self.unet = SingleModel.UNET(n_class)
            self.conv1x1 = tf.keras.layers.Conv2D(n_class, 1, padding="same")

        def call(self, images, training=False):
            output_fpn = self.fpn(images, training)
            output_unet = self.unet(images, training)
            output_final = self.conv1x1(output_fpn * output_unet)
            return output_final

    class FpnUnetSummation(tf.keras.Model):
        def __init__(self, n_class=8):
            super().__init__(name="Fpn_Unet_Summation")

            self.fpn = SingleModel.FPN(n_class)
            self.unet = SingleModel.UNET(n_class)
            self.conv1x1 = tf.keras.layers.Conv2D(n_class, 1, padding="same")

        def call(self, images, training=False):
            output_fpn = self.fpn(images, training)
            output_unet = self.unet(images, training)
            output_final = self.conv1x1(output_fpn + output_unet)
            return output_final

    class FpnUnetConcatenation(tf.keras.Model):
        def __init__(self, n_class=8):
            super().__init__(name="Fpn_Unet_Concatenation")

            self.fpn = SingleModel.FPN(n_class)
            self.unet = SingleModel.UNET(n_class)
            self.conv1x1 = tf.keras.layers.Conv2D(n_class, 1, padding="same")
            self.concatenation = tf.keras.layers.Concatenate()

        def call(self, images, training=False):
            output_fpn = self.fpn(images, training)
            output_unet = self.unet(images, training)
            output = self.concatenation([output_fpn, output_unet])
            output_final = self.conv1x1(output)
            return output_final

    # FPN + FCN
    class FpnFcnConcatenation(tf.keras.Model):
        def __init__(self, n_class=8):
            super().__init__(name="Fpn_Fcn_Concatenation")

            self.fpn = SingleModel.FPN(n_class)
            self.fcn = SingleModel.FCN(n_class)
            self.conv1x1 = tf.keras.layers.Conv2D(n_class, 1, padding="same")
            self.concatenation = tf.keras.layers.Concatenate()

        def call(self, images, training=False):
            output_fpn = self.fpn(images, training)
            output_fcn = self.fcn(images, training)
            output = self.concatenation([output_fpn, output_fcn])
            output_final = self.conv1x1(output)
            return output_final


class RescalingUnet(tf.keras.layers.Layer):
    def __init__(self):
        super(RescalingUnet, self).__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def call(self, inputs):
        return ((inputs * (1 / 255.0)) - self.mean) / self.std
