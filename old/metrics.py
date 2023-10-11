import tensorflow as tf


def jindex_class(target, pred):
    '''
    Range of target and pred must be [0, 1]
    Must be logits
    Make sure input dimension is [B, H, W, C]
    Should return [B, C]
    '''
    intersection = tf.reduce_sum(target * pred, axis=[1, 2])
    union = tf.reduce_sum(target + pred, axis=[1, 2]) - intersection
    return (intersection + 1e-9) / (union + 1e-9)


def dice_coef(target, pred):
    '''
    Range of target and pred must be [0, 1]
    Must be logits
    Make sure input dimension is [B, H, W, C]
    Should return [B, C]
    '''
    nominator = 2 * tf.reduce_sum(target * pred, axis=[1, 2])
    denominator = tf.reduce_sum(target + pred, axis=[1, 2])
    return (nominator + 1e-9) / (denominator + 1e-9)


def dice_loss(target, pred):
    """
    This loss does not reduce batch dimension
    Therefore, the output os [B, 1]
    """
    return 1 - tf.reduce_mean(dice_coef(target, pred), axis=-1)



