import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import re


def plot_confusion(y_true, y_pred, labels, out_png: str):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return cm


def decode_rgb(path: str) -> np.ndarray:
    b = tf.io.read_file(path)
    x = tf.image.decode_image(b, channels=3, expand_animations=False)
    return (tf.cast(x, tf.float32) / 255.0).numpy()


def visual_sanity_check(df_orig, df_show, show_mode):
    orig_lookup = {(r.split, r.label, r.base): r.filepath for _, r in df_orig.iterrows()}

    def base_core_from_export(path: str) -> str:
        """Remove _masked/_crop suffix from exported filename stem."""
        stem = Path(path).stem
        stem = re.sub(r"_(masked|crop)$", "", stem, flags=re.IGNORECASE)
        return stem

    samples = df_show.sample(n=min(6, len(df_show)))

    for _, row in samples.iterrows():
        # find original path robustly
        core = base_core_from_export(row.filepath)
        orig_path = orig_lookup.get((row.split, row.label, core))
        if orig_path is None:
            # fallback: ignore label; try any row in same split with matching base
            cand = df_orig[(df_orig["split"] == row.split) & (df_orig["base"] == core)]
            if len(cand):
                orig_path = cand["filepath"].iloc[0]
            else:
                print(f"[WARN] Could not find original for: {row.filepath} (core='{core}') — skipping")
                continue

        # read images
        orig = decode_rgb(orig_path)
        exp  = np.array(Image.open(row.filepath))
        if exp.ndim == 2:  # grayscale → RGB for display
            exp = np.repeat(exp[..., None], 3, axis=-1)

        # show
        plt.figure(figsize=(7, 3))
        plt.subplot(1, 2, 1); plt.imshow(orig); plt.title(f"Original\n{row.label}"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(exp);  plt.title(show_mode.capitalize()); plt.axis("off")
        plt.tight_layout(); plt.show()
