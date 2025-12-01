from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

def unet_mobilenetv2(input_shape=(128, 128, 3), base_trainable=False):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = base_trainable
    skips = [
        base.get_layer(name).output
        for name in ["block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu"]
    ]
    x = base.get_layer("block_16_project").output  # bottleneck

    up_filters = [512, 256, 128, 64]
    for sk, f in zip(reversed(skips), up_filters):
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, sk])
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)

    # Final up to match input spatial dims
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return Model(inputs=base.input, outputs=out, name="unet_mobilenetv2")
