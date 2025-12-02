# Assert that the image filename and mask filename are similar for all rows
from difflib import SequenceMatcher
from pathlib import Path


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
