import argparse, os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from src.utils.seed import set_global_seed
from src.utils.metrics_seg import dice_coef, iou
from src.seg.unet_mobilenetv2 import unet_mobilenetv2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def load_image_mask(img_path, msk_path, img_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0

    # Mask: decode with 3 channels (BMP-safe), convert to grayscale, then binarize
    msk = tf.io.read_file(msk_path)
    msk = tf.image.decode_image(msk, channels=3, expand_animations=False)  # BMP/PNG/JPG safe
    msk = tf.image.rgb_to_grayscale(msk)
    msk = tf.image.resize(msk, (img_size, img_size), method="nearest")
    msk = tf.cast(msk > 127, tf.float32)

    return img, msk

def augment(img, msk):
    if tf.random.uniform(()) < 0.5:
        img = tf.image.flip_left_right(img)
        msk = tf.image.flip_left_right(msk)
    if tf.random.uniform(()) < 0.5:
        img = tf.image.flip_up_down(img)
        msk = tf.image.flip_up_down(msk)
    return img, msk

def make_ds(df, split, img_size, batch, augmenting):
    sub = df[df["split"] == split]
    ds = tf.data.Dataset.from_tensor_slices((sub["image_path"].values, sub["mask_path"].values))
    ds = ds.shuffle(len(sub), reshuffle_each_iteration=True) if split == "train" else ds
    ds = ds.map(lambda i, m: load_image_mask(i, m, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    if augmenting:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_global_seed(args.seed)

    df = pd.read_csv(args.csv)
    train_ds = make_ds(df, "train", args.img, args.batch, augmenting=True)
    val_ds   = make_ds(df, "val",   args.img, args.batch, augmenting=False)

    model = unet_mobilenetv2(input_shape=(args.img, args.img, 3), base_trainable=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="binary_crossentropy",
                  metrics=["binary_accuracy", dice_coef, iou])

    cbs = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(os.path.join(args.out, "best.keras"), save_best_only=True, monitor="val_loss"),
        CSVLogger(os.path.join(args.out, "history.csv"))
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)
    model.save(os.path.join(args.out, "last.keras"))

if __name__ == "__main__":
    main()
