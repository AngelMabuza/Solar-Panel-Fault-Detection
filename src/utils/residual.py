from tensorflow.keras import layers

def conv_bn_lrelu(x, filters, k: int = 3, s: int = 1):
    y = layers.Conv2D(filters, k, strides=s, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    return y

def residual_block(x, filters):
    y = conv_bn_lrelu(x, filters)
    y = conv_bn_lrelu(y, filters)
    if int(x.shape[-1]) != filters:
        skip = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        skip = layers.BatchNormalization()(skip)
    else:
        skip = x
    return layers.Add()([skip, y])
