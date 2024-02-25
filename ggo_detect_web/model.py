import tensorflow as tf

from config import lggo_input_size


def conv_block(inputs, filters, dimension=2):
    if dimension not in [2, 3]:
        raise ValueError("The 'dimension' parameter must be either 2 or 3.")

    conv_func = tf.keras.layers.Conv2D if dimension == 2 else tf.keras.layers.Conv3D
    conv = conv_func(filters, kernel_size=1 if dimension == 3 else (3, 3), padding='same')(inputs)
    norm = tf.keras.layers.BatchNormalization()(conv)
    activation = tf.keras.layers.LeakyReLU(alpha=0.2)(norm)
    return activation


def attention_block(gate, skip_connection, n_coefficients, dimension=2):
    if dimension not in [2, 3]:
        raise ValueError("The 'dimension' parameter must be either 2 or 3.")

    def conv_bn(x, filters):
        x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=True)(
            x) if dimension == 2 else tf.keras.layers.Conv3D(filters, kernel_size=1, strides=1, padding='same',
                                                             use_bias=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    W_gate = conv_bn(gate, n_coefficients)
    W_x = conv_bn(skip_connection, n_coefficients)

    psi = tf.keras.layers.Activation('relu')(W_gate + W_x)
    psi = tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(
        psi) if dimension == 2 else tf.keras.layers.Conv3D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(
        psi)
    psi = tf.keras.layers.BatchNormalization()(psi)
    psi = tf.keras.layers.Activation('sigmoid')(psi)

    out = skip_connection * psi
    return out


def downsample_block(inputs, filters, dimension=2):
    conv1 = conv_block(inputs, filters)
    conv2 = conv_block(conv1, filters)
    downsample = tf.keras.layers.MaxPooling2D(pool_size=2)(conv2) if dimension == 2 else tf.keras.layers.MaxPooling3D(
        pool_size=(2, 2, 1))(conv2)
    return conv2, downsample


def upsample_block(inputs, skip_connection, filters, dimension=2, lung=False):
    upsample = tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(
        inputs) if dimension == 2 else tf.keras.layers.Conv3DTranspose(filters, kernel_size=2, strides=(2, 2, 1),
                                                                       padding='same')(
        inputs)

    if not lung:
        attention = attention_block(upsample, skip_connection, filters, dimension)

        if dimension == 2:
            conv1 = conv_block(attention, filters, dimension)
        else:
            concat = tf.keras.layers.Concatenate()([attention, upsample])
            conv1 = conv_block(concat, filters, dimension)

    else:
        # concat = tf.keras.layers.Concatenate()([skip_connection, upsample])
        conv1 = conv_block(upsample, filters, dimension)
    conv2 = conv_block(conv1, filters, dimension)
    return conv2


def build_unet(input_shape, chanel=16, stage=5, dimension=2):
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    conv = []
    encoder_layers = []

    # Stage 1
    for i in range(1, stage):
        conv1 = conv_block(x, chanel * 2 ** (i - 1))
        conv2 = conv_block(conv1, chanel * 2 ** (i - 1))
        encoder_layers.append(conv2)
        encoder_layers.append(conv1)
        downsample, x = downsample_block(conv2, chanel * 2 ** (i - 1))
        conv.append(conv2)

    # Stage 5
    conv_last_1 = conv_block(x, chanel * 2 ** (stage - 1))
    encoder_layers.append(conv_last_1)
    x = conv_block(conv_last_1, chanel * 2 ** (stage - 1))

    for layer in encoder_layers:
        layer.trainable = False

    # Upsampling
    for i in range(1, stage):
        x = upsample_block(x, conv[stage - i - 1], chanel * 2 ** (stage - i - 1), dimension, True)

    # Final convolution
    x = tf.keras.layers.Conv2D(1, 1)(x)
    outputs = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def build_encoder(inputs, channel, stage, dimension=2):
    x = inputs
    conv = []
    encoder_layers = []

    # Stage 1
    for i in range(1, stage):
        conv1 = conv_block(x, channel * 2 ** (i - 1), dimension)
        conv2 = conv_block(conv1, channel * 2 ** (i - 1), dimension)
        encoder_layers.append(conv2)
        encoder_layers.append(conv1)
        downsample, x = downsample_block(conv2, channel * 2 ** (i - 1), dimension)
        conv.append(conv2)

    # Stage 5
    conv_last_1 = conv_block(x, channel * 2 ** (stage - 1), dimension)
    x = conv_block(conv_last_1, channel * 2 ** (stage - 1), dimension)

    return tf.keras.Model(inputs=inputs, outputs=x), conv


def build_decoder(inputs, encoder_layer, channel, stage, dimension=2):
    x = inputs
    conv = encoder_layer

    # Upsampling
    for i in range(1, stage):
        x = upsample_block(x, conv[stage - i - 1], channel * 2 ** (stage - i - 1), dimension)

    # Final convolution
    x = tf.keras.layers.Conv2D(1, 1)(x) if dimension == 2 else tf.keras.layers.Conv3D(1, 1)(x)
    outputs = tf.keras.layers.Activation("sigmoid")(x)

    return outputs


class SGGO_Segment:
    def __init__(self, input_shape, channel=16, stage=5):
        inputs = tf.keras.Input(shape=input_shape)
        self.encoder, layer = build_encoder(inputs, channel, stage)
        self.decoder = build_decoder(self.encoder.outputs[0], layer, channel, stage)
        self.unet = tf.keras.Model(inputs=inputs, outputs=self.decoder)


class LGGO_Segment:
    def __init__(self, input_shape, channel=16, stage=5):
        inputs = tf.keras.Input(shape=input_shape)
        self.encoder, layer = build_encoder(inputs, channel, stage, 3)
        self.decoder = build_decoder(self.encoder.outputs[0], layer, channel, stage, 3)
        self.unet = tf.keras.Model(inputs=inputs, outputs=self.decoder)


# lggo = LGGO_Segment(lggo_input_size, channel=16, stage=5)
# model = lggo.unet
# model.summary()