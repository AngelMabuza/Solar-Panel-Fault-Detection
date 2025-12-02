# src/cls/models/cnn_plain.py
# Minimal plain CNN with the SAME signature as your other builders.
# Accepts (input_shape, num_classes, lr, base_trainable, **kwargs) so the common
# training loop won't crash. If your labels are integers, set labels_are_integers=True.

from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, Model

__all__ = ["build_cnn_plain", "build_model", "build_plain_cnn"]

def build_cnn_plain(
    input_shape=(256, 256, 3),
    num_classes: int = 6,
    lr: float = 1e-4,
    base_trainable: bool = False,      # ignored (for API compatibility)
    labels_are_integers: bool = False, # True if you feed integer labels (not one-hot)
    **kwargs,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="input")
    x = inputs
    # (Optional but helpful) normalize raw images
    # x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)
    # Your original blocks (kept as-is)

    for f in [32, 64, 128]:  # Reduced a block - 256
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)  # New
        x = layers.Activation("relu")(x)  # New
        x = layers.MaxPooling2D()(x)

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, out, name="cnn_plain")

    # Pick the right loss for your label encoding
    loss = (
        "sparse_categorical_crossentropy" if labels_are_integers
        else "categorical_crossentropy"
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model


def build_model(**kwargs) -> tf.keras.Model:
    return build_cnn_plain(**kwargs)

def build_plain_cnn(**kwargs) -> tf.keras.Model:
    return build_cnn_plain(**kwargs)
