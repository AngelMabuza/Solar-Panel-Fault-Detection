# Assert that the image filename and mask filename are similar for all rows
from difflib import SequenceMatcher
from pathlib import Path
import tensorflow as tf


def check_image_mask_pairs(df):
    """
    Test the similarity between image url vs mask image url
    """
    def filename_similarity(path_a, path_b):
        a = Path(path_a).name
        b = Path(path_b).name
        return SequenceMatcher(None, a, b).ratio()

    img_col = df.columns[1]
    msk_col = df.columns[2]

    THRESH = 0.60
    bad_rows = []
    for idx, row in df.iterrows():
        sim = filename_similarity(row[img_col], row[msk_col])
        if sim < THRESH:
            bad_rows.append((int(idx), row[img_col], row[msk_col], float(sim)))

    if bad_rows:
        print(f"Found {len(bad_rows)} rows with filename similarity below {THRESH:.2f}:")
        for r in bad_rows[:20]:
            print(f"Row {r[0]}: {r[1]} <-> {r[2]} (sim={r[3]:.3f})")
        # Fail loudly to stop the notebook if many rows are mismatched
        assert False, f"Filename similarity too low for {len(bad_rows)} rows (threshold {THRESH:.2f}). See printed rows for examples."
    else:
        return print(f"All filenames for image path and mask path pairs are aligned (similarity >= {THRESH:.2f}).")


def load_image_mask2(img_path, msk_path, img_size):
    # Image: OK to decode with 3 channels
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)  # BMP/PNG/JPG
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0

    # Mask: decode with 3 channels (works for BMP/PNG/JPG), then convert to grayscale
    msk = tf.io.read_file(msk_path)
    msk = tf.image.decode_image(msk, channels=3, expand_animations=False)  # <-- changed from channels=1
    msk = tf.image.rgb_to_grayscale(msk)                                   # <-- ensure single channel
    msk = tf.image.resize(msk, (img_size, img_size), method="nearest")
    msk = tf.cast(msk > 127, tf.float32)                                   # binarize
    return img.numpy(), msk.numpy()


def load_image_mask(img_path, msk_path, img_size,
                    thr=0.5, rescue_thr=0.15,  # main + fallback thresholds
                    auto_invert=True, dilate_px=1):
    # ----- IMAGE -----
    img_b = tf.io.read_file(img_path)
    img   = tf.image.decode_image(img_b, channels=3, expand_animations=False)
    img   = tf.image.resize(img, (img_size, img_size), method="bilinear")
    img   = tf.cast(img, tf.float32) / 255.0

    # ----- MASK (robust to tiny/far panels) -----
    mb = tf.io.read_file(msk_path)
    m  = tf.image.decode_image(mb, channels=3, expand_animations=False)
    m  = tf.image.rgb_to_grayscale(m)               # (H,W,1), uint8
    m  = tf.cast(m, tf.float32) / 255.0             # -> [0,1]

    if auto_invert:
        mean_val = tf.reduce_mean(m)
        m = tf.where(mean_val > 0.7, 1.0 - m, m)    # flip if mostly white

    # Downscale with AREA to preserve tiny positives
    m_small = tf.image.resize(m, (img_size, img_size),
                              method=tf.image.ResizeMethod.AREA)

    # Threshold
    m_bin = tf.cast(m_small >= thr, tf.float32)

    # Rescue path if empty: lower thr + 1-px dilation
    if tf.reduce_sum(m_bin) == 0:
        m_bin = tf.cast(m_small >= rescue_thr, tf.float32)
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            m_bin = tf.nn.max_pool2d(
                m_bin[None, ...], ksize=[1, k, k, 1], strides=[1, 1, 1, 1], padding="SAME"
            )[0]

    # Ensure binary and **squeeze to (H,W)**
    m_bin = tf.cast(m_bin > 0.5, tf.float32)        # (H,W,1)
    m_bin = tf.squeeze(m_bin, axis=-1)              # (H,W)

    return img, m_bin
