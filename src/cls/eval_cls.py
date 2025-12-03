import argparse, os, json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from src.utils.plotting import plot_confusion

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img", type=int, default=256)
    return ap.parse_args()

def load_image(fp, img_size):
    img = tf.io.read_file(fp)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv, sep=';')  # added sep param
    labels = sorted(df["label"].unique())
    enc = LabelEncoder().fit(labels)

    test_df = df[df["split"]=="test"].copy()
    y_true = enc.transform(test_df["label"].values)

    model = tf.keras.models.load_model(args.model, compile=False)
    # Predict
    preds = []
    for fp in test_df["filepath"].values:
        img = load_image(fp, args.img)
        p = model(tf.expand_dims(img, 0), training=False).numpy()[0]
        preds.append(np.argmax(p))
    y_pred = np.array(preds)

    # Report
    report = classification_report(y_true, y_pred, target_names=enc.classes_, zero_division=0, digits=3, output_dict=True)
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    plot_confusion(y_true, y_pred, labels=enc.classes_.tolist(),
                   out_png=os.path.join(args.out, "confusion_matrix.png"))
    print("Saved metrics and confusion matrix.")

if __name__ == "__main__":
    main()
