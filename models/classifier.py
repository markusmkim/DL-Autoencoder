from tensorflow.keras import layers, Input, Model


def deep_classifier(number_of_classes, input_shape=None, encoder=None):
    if not input_shape and not encoder:
        return None
    if not input_shape:
        input_shape = encoder.input_shape[1:-1]

    input_layer = Input(shape=input_shape + (1, ))
    if encoder:
        layer = encoder(input_layer)
    else:
        # start with convolutional layers
        layer = layers.Conv2D(4, (3, 3), padding="valid", activation="relu")(input_layer)
        layer = layers.Conv2D(6, (3, 3), padding="valid", activation="relu")(layer)
        layer = layers.Flatten()(layer)
        layer = layers.Dense(256, activation="relu")(layer)
    layer = layers.Dense(128, activation="relu")(layer)
    output_layer = layers.Dense(number_of_classes, activation="softmax")(layer)
    return Model(input_layer, output_layer)