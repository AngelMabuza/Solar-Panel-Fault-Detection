import argparse
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def load_image_mask(img_path, msk_path, img_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    msk = tf.io.read_file(msk_path)
    msk = tf.image.decode_image(msk, channels=3, expand_animations=False)  # BMP/PNG/JPG safe
    msk = tf.image.rgb_to_grayscale(msk)
    msk = tf.image.resize(msk, (img_size, img_size), method="nearest")
    msk = tf.cast(msk > 127, tf.float32)
    return img.numpy(), msk.numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sub = df[df["split"] == "train"].sample(n=min(args.n, len(df)))
    for _, r in sub.iterrows():
        img, msk = load_image_mask(r.image_path, r.mask_path, args.img)
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(img); ax[0].set_title("Image"); ax[0].axis("off")
        ax[1].imshow(msk.squeeze(), cmap="gray"); ax[1].set_title("Mask"); ax[1].axis("off")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
