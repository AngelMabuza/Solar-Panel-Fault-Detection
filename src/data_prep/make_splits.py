import argparse, os, random, csv
from pathlib import Path
from collections import defaultdict

ALLOWED_IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp"]
ALLOWED_MSK_EXTS_DEFAULT = [".png", ".jpg", ".jpeg", ".bmp"]

def stratified_split(items_by_class, seed=42, train=0.6, val=0.2, test=0.2):
    random.seed(seed)
    out = {"train": [], "val": [], "test": []}
    for cls, items in items_by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train)
        n_val = int(n * val)
        train_items = items[:n_train]
        val_items = items[n_train:n_train+n_val]
        test_items = items[n_train+n_val:]
        out["train"] += [(cls, x) for x in train_items]
        out["val"] += [(cls, x) for x in val_items]
        out["test"] += [(cls, x) for x in test_items]
    return out

def parse_exts(exts_str, defaults):
    if not exts_str:
        return defaults
    return [e.strip().lower() for e in exts_str.split(",") if e.strip()]

def find_seg_pairs_same_dir(root: Path, mask_suffix: str, img_exts, msk_exts):
    """
    Recursively walk 'root' and look for pairs where an *image* exists
    and a sibling *mask* whose basename is image_basename + mask_suffix,
    in the same directory. Example: 'PV08_123.bmp' + 'PV08_123_label.bmp'.
    """
    pairs = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        base = p.stem
        # skip masks themselves
        if base.endswith(mask_suffix):
            continue
        if ext not in img_exts:
            continue
        mask_base = base + mask_suffix
        found = None
        for me in msk_exts:
            cand = p.with_name(mask_base + me)
            if cand.exists():
                found = cand
                break
        if found is not None:
            pairs.append((str(p), str(found)))
    return pairs

def find_seg_pairs_classic(root: Path, img_exts, msk_exts):
    """Classic layout: root/images/* and root/masks/* with identical basenames."""
    img_dir = root / "images"
    msk_dir = root / "masks"
    images = sorted([p for p in img_dir.rglob("*.*") if p.suffix.lower() in img_exts])
    pairs = []
    for img in images:
        base = img.stem
        found = None
        for me in msk_exts:
            cand = msk_dir / (base + me)
            if cand.exists():
                found = cand
                break
        if found is None:
            cands = list(msk_dir.glob(base + ".*"))
            if cands:
                found = cands[0]
        if found is not None:
            pairs.append((str(img), str(found)))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cls", "seg"], required=True)
    ap.add_argument("--root", required=True, help="Root folder for dataset")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--seed", type=int, default=42)

    # Segmentation options
    ap.add_argument("--seg_same_dir", action="store_true",
                    help="Images and masks live in the same folders with mask_suffix appended to image basename.")
    ap.add_argument("--mask_suffix", default="_label",
                    help="Suffix appended to image basename to get mask basename (e.g., _label).")
    ap.add_argument("--img_exts", default=None,
                    help="Comma-separated allowed image extensions (e.g., .bmp,.png,.jpg).")
    ap.add_argument("--mask_exts", default=None,
                    help="Comma-separated allowed mask extensions (e.g., .bmp,.png,.jpg).")

    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "cls":
        # Expect: root/class_name/*
        items_by_class = defaultdict(list)
        for cls_dir in sorted(Path(args.root).glob("*")):
            if not cls_dir.is_dir(): 
                continue
            cls = cls_dir.name
            for img in sorted(cls_dir.rglob("*.*")):
                items_by_class[cls].append(str(img))
        splits = stratified_split(items_by_class, seed=args.seed)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["split", "filepath", "label"])
            for split in ["train", "val", "test"]:
                for cls, fp in splits[split]:
                    w.writerow([split, fp, cls])

    else:
        img_exts = parse_exts(args.img_exts, ALLOWED_IMG_EXTS_DEFAULT)
        msk_exts = parse_exts(args.mask_exts, ALLOWED_MSK_EXTS_DEFAULT)
        root = Path(args.root)

        if args.seg_same_dir:
            pairs = find_seg_pairs_same_dir(root, args.mask_suffix, img_exts, msk_exts)
        else:
            pairs = find_seg_pairs_classic(root, img_exts, msk_exts)

        if len(pairs) == 0:
            raise SystemExit("No (image, mask) pairs found. Check --root and suffix/ext options.")

        # single-class split (binary segmentation)
        items_by_class = {"mask": pairs}
        splits = stratified_split(items_by_class, seed=args.seed)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["split", "image_path", "mask_path"])
            for split in ["train", "val", "test"]:
                for _, (img, msk) in splits[split]:
                    w.writerow([split, img, msk])

if __name__ == "__main__":
    main()
