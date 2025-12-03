import os
import csv
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def is_image_file(fname):
    return os.path.splitext(fname)[1].lower() in IMAGE_EXTS


def count_images(base_dir='data/classification/images'):
    counts = OrderedDict()
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    for entry in sorted(os.listdir(base_dir)):
        cls_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(cls_dir):
            continue
        cnt = 0
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if is_image_file(f):
                    cnt += 1
        counts[entry] = cnt
    return counts


def save_counts_csv(counts, out_csv='reports/class_counts.csv'):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'count'])
        for cls, cnt in counts.items():
            writer.writerow([cls, cnt])


def plot_counts(counts, out_png='reports/class_distribution.png'):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    classes = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, values, color='tab:blue')
    ax.set_ylabel('Number of images')
    ax.set_xlabel('Class')
    ax.set_title('Image count per class')
    plt.xticks(rotation=45, ha='right')

    # annotate counts on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, str(val), ha='center', va='bottom')

    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    try:
        counts = count_images()
    except FileNotFoundError as e:
        print(e)
        return 1

    # print table
    print('Class counts:')
    for cls, cnt in counts.items():
        print(f'{cls}: {cnt}')

    save_counts_csv(counts)
    plot_counts(counts)
    print('\nSaved CSV to `reports/class_counts.csv` and plot to `reports/class_distribution.png`.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
