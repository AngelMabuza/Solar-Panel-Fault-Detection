import argparse, os, json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import LabelEncoder
from src.utils.seed import set_global_seed

# Model registry
from src.cls.models.inceptionv3_net_proposed import build_inceptionv3_net_proposed
from src.cls.models.inceptionv3_base import build_inceptionv3_base
from src.cls.models.resnet50 import build_resnet50
from src.cls.models.vgg16 import build_vgg16
from src.cls.models.vgg19 import build_vgg19
from src.cls.models.densenet import build_densenet
from src.cls.models.mobilenetv3 import build_mobilenetv3
from src.cls.models.cnn_plain import build_cnn_plain

REGISTRY = {
    "inceptionv3_net_proposed": build_inceptionv3_net_proposed,
    "inceptionv3_base": build_inceptionv3_base,
    "resnet50": build_resnet50,
    "vgg16": build_vgg16,
    "vgg19": build_vgg19,
    "densenet": build_densenet,
    "mobilenetv3": build_mobilenetv3,
    "cnn_plain": build_cnn_plain,
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model_name", choices=list(REGISTRY.keys()), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_image(fp, img_size):
    img = tf.io.read_file(fp)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32)/255
    return img


def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img

def make_ds(df, split, encoder, img_size, batch, augmenting):
    sub = df[df["split"] == split].copy()
    y = encoder.transform(sub["label"].values)
    y = to_categorical(y, num_classes=len(encoder.classes_))
    ds_x = tf.data.Dataset.from_tensor_slices(sub["filepath"].values)
    ds_y = tf.data.Dataset.from_tensor_slices(y.astype(np.float32))
    ds = tf.data.Dataset.zip((ds_x, ds_y))
    if split == "train":
        ds = ds.shuffle(len(sub), reshuffle_each_iteration=True)
    ds = ds.map(lambda fp, yy: (load_image(fp, img_size), yy), num_parallel_calls=tf.data.AUTOTUNE)
    if augmenting:
        ds = ds.map(lambda im, yy: (augment(im), yy), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_global_seed(args.seed)
    try:
        df = pd.read_csv(args.csv, sep=',')
    except:
        df = pd.read_csv(args.csv, sep=';')
    encoder = LabelEncoder()
    encoder.fit(sorted(df["label"].unique()))
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(encoder.classes_.tolist(), f, indent=2)

    train_ds = make_ds(df, "train", encoder, args.img, args.batch, augmenting=True)
    val_ds   = make_ds(df, "val",   encoder, args.img, args.batch, augmenting=False)

    build_fn = REGISTRY[args.model_name]
    model = build_fn(input_shape=(args.img, args.img, 3), num_classes=len(encoder.classes_), lr=args.lr, base_trainable=False)

    cbs = [
        EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(os.path.join(args.out, "best.keras"), save_best_only=True, monitor="val_accuracy"),
        CSVLogger(os.path.join(args.out, "history.csv"))
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)
    model.save(os.path.join(args.out, "last.keras"))
 

if __name__ == "__main__":
    main()
