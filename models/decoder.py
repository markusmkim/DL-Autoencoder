from tensorflow.keras import layers, Input, Model


def deep_conv_decoder(input_dim, output_shape):
    input_layer = Input(shape=(input_dim, ))
    layer = layers.Dense(256, activation="relu")(input_layer)
    layer = layers.Dense((output_shape[0] - 4) * (output_shape[1] - 4) * 6, activation="relu")(layer)
    layer = layers.Reshape(((output_shape[0] - 4), (output_shape[1] - 4), 6))(layer)
    layer = layers.Conv2DTranspose(4, (3, 3), padding="valid", activation="relu")(layer)
    output_layer = layers.Conv2DTranspose(1, (3, 3), padding="valid", activation="relu")(layer)
    # print(layer.shape)
    # return layer
    return Model(input_layer, output_layer)