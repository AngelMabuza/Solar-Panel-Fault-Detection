import tensorflow as tf


class VGG16:
    def __init__(self, img_height, img_width, train_ds, val_ds, num_classes=90):
        self.img_height = img_height
        self.img_width = img_width
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.num_classes = num_classes
        # Build and store the model instance on initialization
        self.model = self._build_model()

    def _build_model(self):
        base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3)
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(self.img_height, self.img_width, 3))
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

    def train_model(self, epochs):
        if not hasattr(self, 'model') or self.model is None:
            # if for some reason the model wasn't built during init, build it now
            self.model = self._build_model()

        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-2,
                    patience=3,
                    verbose=1,
                    restore_best_weights=True
                )
            ],
        )
        # save history and return it
        self.history = history
        return history
