from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19
from src.utils.se_block import se_block
from src.utils.residual import residual_block

def build_vgg19(input_shape=(256, 256, 3), num_classes=6, lr=1e-4, base_trainable=False):
    base = VGG19(include_top=False, input_shape=input_shape, weights="imagenet")
    base.trainable = base_trainable
    x = base.output
    x = residual_block(x, 512)
    x = se_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out, name="vgg19")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
