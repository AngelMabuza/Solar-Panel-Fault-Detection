import argparse, os, json
import pandas as pd
import numpy as np
import tensorflow as tf
from src.utils.metrics_seg import dice_coef, iou

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img", type=int, default=128)
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

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    sub = df[df["split"] == "test"]
    model = tf.keras.models.load_model(args.model, custom_objects={"dice_coef": dice_coef, "iou": iou})

    # Evaluate
    accs, dices, ious = [], [], []
    for _, row in sub.iterrows():
        img, msk = load_image_mask(row.image_path, row.mask_path, args.img)
        pred = model(tf.expand_dims(img, 0), training=False)
        accs.append(tf.keras.metrics.binary_accuracy(msk, tf.squeeze(pred, 0)).numpy().mean())
        dices.append(dice_coef(tf.expand_dims(msk, 0), pred).numpy())
        ious.append(iou(tf.expand_dims(msk, 0), pred).numpy())

    metrics = {
        "binary_accuracy": float(np.mean(accs)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "n_test": int(len(sub))
    }
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

if __name__ == "__main__":
    main()
