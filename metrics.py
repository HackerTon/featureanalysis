import tensorflow as tf


# Mode must OUTPUT in logits [-inf, inf]
# Make sure input dimension is [B, H, W, C]
def jindex_class(target, pred):
    intersection = tf.reduce_sum(target * pred, axis=[1, 2])
    union = tf.reduce_sum(target + pred, axis=[1, 2]) - intersection
    return tf.reduce_mean((intersection + 1e-9) / (union + 1e-9), axis=0)


def dice_coef(target, pred):
    nominator = 2 * tf.reduce_sum(target * pred, axis=[1, 2])
    denominator = tf.reduce_sum(target + pred, axis=[1, 2])
    return tf.reduce_mean((nominator + 1e-9) / (denominator + 1e-9), axis=0)


def dice_loss(target, pred):
    """
    This loss does not reduce batch dimension
    Therefore, the output os [B, 1]
    """
    return 1 - dice_coef(target, pred)
