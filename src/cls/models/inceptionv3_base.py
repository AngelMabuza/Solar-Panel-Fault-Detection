# src/cls/models/inceptionv3_base.py
# Simple "base" head on top of InceptionV3 (ImageNet backbone)
# Matches the style of your other builders (compile inside, same args).

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


def build_inceptionv3_base(
    input_shape=(256, 256, 3),
    num_classes=6,
    lr=1e-4,
    base_trainable=False,
):
    """
    Build an InceptionV3-based classifier with a lightweight head.

    Args:
        input_shape:   (H, W, 3)
        num_classes:   number of classes for softmax output
        lr:            learning rate for Adam
        base_trainable: if True, unfreezes the backbone (fine-tuning)

    Returns:
        Compiled tf.keras.Model
    """
    # Backbone
    base = InceptionV3(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = base_trainable

    # Input & preprocessing (use official preprocess_input for InceptionV3)
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(preprocess_input, name="preprocess")(inputs)  # expects 0..255; raw images are typically 0..255

    # Features
    x = base(x, training=False)  # keep BN in inference mode when frozen
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Lightweight head (LeakyReLU + BN + Dropout)
    x = layers.Dense(256, name="fc1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="lrelu1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.30, name="drop1")(x)

    x = layers.Dense(128, name="fc2")(x)
    x = layers.LeakyReLU(alpha=0.1, name="lrelu2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(0.30, name="drop2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = Model(inputs=inputs, outputs=outputs, name="inceptionv3_base")

    # Compile (match your other models)
    # NOTE: If your labels are integer-encoded, switch to "sparse_categorical_crossentropy".
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

