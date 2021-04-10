import tensorflow as tf


def optimizer(name, learning_rate):
    """
    Returns optimizer with given name.
    If name is invalid, function will return SGD as default.
    Valid options: adagrad | rmsprop | adam | sgd
    """
    if name == 'adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    if name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return tf.keras.optimizers.SGD(learning_rate=learning_rate)
