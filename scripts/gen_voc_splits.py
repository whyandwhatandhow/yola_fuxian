#!/usr/bin/env python3
"""Generate train/val/test text files for VOC-style dataset.

Usage:
    python gen_voc_splits.py /path/to/dataset [--train 0.8] [--val 0.2] [--test 0.0]

The dataset directory should contain `JPEGImages/` (or `images/`) and/or `Annotations/`.
Each text file will list the base names (no extension), one per line.  The script
randomly shuffles the sample list before splitting.

Example:
    python gen_voc_splits.py data/darkface --train 0.75 --val 0.25


"""
import argparse
import os
import random

def main():
    parser = argparse.ArgumentParser(description="Generate VOC train/val/test lists")
    parser.add_argument("dataset_dir", help="root of dataset containing JPEGImages or Annotations")
    parser.add_argument("--train", type=float, default=0.8, help="fraction for train split")
    parser.add_argument("--val", type=float, default=0.2, help="fraction for val split")
    parser.add_argument("--test", type=float, default=0.0, help="fraction for test split")
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    args = parser.parse_args()

    ratios = args.train + args.val + args.test
    if abs(ratios - 1.0) > 1e-6:
        parser.error("Sum of train/val/test ratios must equal 1.0")

    # find ids using annotation files if exist, else JPEGImages
    ann_dir = os.path.join(args.dataset_dir, "Annotations")
    img_dir = os.path.join(args.dataset_dir, "JPEGImages")
    ids = []
    if os.path.isdir(ann_dir):
        for f in os.listdir(ann_dir):
            if f.endswith('.xml'):
                ids.append(os.path.splitext(f)[0])
    elif os.path.isdir(img_dir):
        for f in os.listdir(img_dir):
            if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']:
                ids.append(os.path.splitext(f)[0])
    else:
        raise RuntimeError(f"Neither Annotations nor JPEGImages found in {args.dataset_dir}")

    ids = sorted(set(ids))
    random.seed(args.seed)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    def write_list(fn, lst):
        if not lst:
            return
        path = os.path.join(args.dataset_dir, fn)
        with open(path, 'w') as f:
            for x in lst:
                f.write(x + '\n')
        print(f"Wrote {len(lst)} entries to {path}")

    write_list('train.txt', train_ids)
    write_list('val.txt', val_ids)
    write_list('test.txt', test_ids)

if __name__ == '__main__':
    main()
