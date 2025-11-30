import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

class ImageProcessing:
    def __init__(self, train_ds, test_ds, validation_ds):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.validation_ds = validation_ds

    def process_image_splits(self):
        train_ds_size = tf.data.experimental.cardinality(self.train_ds).numpy()
        test_ds_size = tf.data.experimental.cardinality(self.test_ds).numpy()
        validation_ds_size = tf.data.experimental.cardinality(self.validation_ds).numpy()

        print("Dataset sizes:")
        print(f"Training batches: {train_ds_size}")
        print(f"Validation batches: {validation_ds_size}")
        print(f"Test batches: {test_ds_size}\n")

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = (self.train_ds
                    .map(self.process_images, num_parallel_calls=AUTOTUNE)
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=16, drop_remainder=False)
                    .prefetch(AUTOTUNE))
        test_ds = (self.test_ds
                   .map(self.process_images, num_parallel_calls=AUTOTUNE)
                   .shuffle(buffer_size=test_ds_size)
                   .batch(batch_size=16, drop_remainder=False)
                   .prefetch(AUTOTUNE))
        validation_ds = (self.validation_ds
                         .map(self.process_images, num_parallel_calls=AUTOTUNE)
                         .shuffle(buffer_size=validation_ds_size)
                         .batch(batch_size=16, drop_remainder=False)
                         .prefetch(AUTOTUNE))
        return train_ds, test_ds, validation_ds

    def process_images(self, image, label):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize(image, (299, 299))
        return image, label


class LoadImages:

    def __init__(self, directory):
        self.directory = directory
        self.img_size = (299, 299)

    def load_data_from_directory(self):
        images = []
        labels = []
        class_names = []
        for class_idx, class_dir in enumerate(sorted(os.listdir(self.directory))):
            class_path = os.path.join(self.directory, class_dir)
            if not os.path.isdir(class_path):
                continue
            class_names.append(class_dir)
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
                        img_array = keras.preprocessing.image.img_to_array(img)
                        images.append(img_array)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")   
        return np.array(images), np.array(labels), class_names
