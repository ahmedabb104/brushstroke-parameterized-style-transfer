#!/usr/bin/env python3
"""Sanity-check deception assets, label file, and one forward pass."""

import argparse
import os
import sys

from deception_score import (
    compute_deception_rate,
    get_artist_labels,
    get_deception_paths,
    load_deception_model,
    preprocess_image,
)


def _autodetect_style_sample(repo_root: str):
    """
    Pick a painting under images/style/ when --sample is not given.
    Tries common filenames first, then any image in that folder.
    """
    rel_candidates = [
        "images/style/starry_night-van-gogh.jpg",
        "images/style/starry_night.jpg",
        "images/style/still-life-cezanne.jpg",
        "images/style/water-lilies-monet.jpg",
    ]
    for rel in rel_candidates:
        p = os.path.join(repo_root, rel)
        if os.path.isfile(p):
            return p

    style_dir = os.path.join(repo_root, "images", "style")
    if os.path.isdir(style_dir):
        for name in sorted(os.listdir(style_dir)):
            low = name.lower()
            if low.endswith((".jpg", ".jpeg", ".png", ".webp")):
                return os.path.join(style_dir, name)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify deception-score setup.")
    parser.add_argument(
        "--deception-dir",
        default=None,
        help="Folder with split.hdf5 and model.ckpt-* (default: auto-detect)",
    )
    parser.add_argument(
        "--sample",
        default=None,
        metavar="PATH",
        help=(
            "Style image for a test prediction "
            "(default: auto-pick from images/style/)"
        ),
    )
    parser.add_argument(
        "--artist",
        default="vincent-van-gogh",
        help="Artist slug for sample deception check",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if args.sample is None:
        sample_path = _autodetect_style_sample(repo_root)
    else:
        sample_path = args.sample
        if not os.path.isabs(sample_path):
            sample_path = os.path.join(repo_root, sample_path)

    try:
        root, split_path, ckpt_prefix = get_deception_paths(args.deception_dir)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    print(f"Root: {root}")
    print(f"split.hdf5: {split_path}")
    print(f"Checkpoint: {ckpt_prefix}")

    labels = get_artist_labels(split_path)
    print(f"Unique artist slugs in split: {len(labels)}")

    model = load_deception_model(args.deception_dir)

    if not sample_path or not os.path.isfile(sample_path):
        print(
            "No sample image found under images/style/. "
            "Add a style .jpg or pass --sample PATH; skipping prediction test."
        )
        return

    if args.sample is None:
        print(f"Auto-selected sample: {sample_path}")

    batch = preprocess_image(sample_path)
    preds = model.predict(batch)
    print(f"Argmax class index for sample: {int(preds[0])}")
    inv = {v: k for k, v in labels.items()}
    print(f"Predicted slug (inverse map): {inv.get(int(preds[0]), '(multiple or unknown)')}")

    rate, details = compute_deception_rate(
        model,
        [sample_path],
        target_artist=args.artist,
        split_hdf5_path=split_path,
        artist_labels=labels,
    )
    print(
        f"Deception rate vs '{args.artist}' on sample only: {rate:.2%} "
        f"({details['correct']}/{details['total']})"
    )


if __name__ == "__main__":
    main()
