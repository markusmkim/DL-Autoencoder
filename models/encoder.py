from tensorflow.keras import layers, Input, Model


def deep_conv_encoder(input_shape, output_dim):
    input_layer = Input(shape=input_shape + (1, ))
    layer = layers.Conv2D(4, (3, 3), padding="valid", activation="relu")(input_layer)
    layer = layers.Conv2D(6, (3, 3), padding="valid", activation="relu")(layer)
    pre_flatten_shape = layer.shape
    layer = layers.Flatten()(layer)
    layer = layers.Dense(256, activation="relu")(layer)
    output_layer = layers.Dense(output_dim, activation="relu")(layer)
    # return layer
    return Model(input_layer, output_layer, name="encoder")
