'''
Code to download dataset from AIME 2025 huggingface.
cmd: python download_dataset.py MathArena/aime_2025
'''

import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo_id", help="e.g. MathArena/aime_2025 (copy from the dataset page URL)")
    p.add_argument("--config", default=None, help="subset name, if the dataset has several")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="data", help="output folder")
    args = p.parse_args()
    if args.split == "all":
        args.split = None

    ds =load_dataset(args.repo_id, args.config, split=args.split)
    print(ds)

    out_dir = Path(args.out) / args.repo_id.replace("/", "__")
    if args.config:
        out_dir = out_dir / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {args.split: ds} if args.split else ds
    for name, split_ds in splits.items():
        path = out_dir / f"{name}.jsonl"
        split_ds.to_json(path, lines=True, force_ascii=False)
        print(f"Saved {len(split_ds)} rows -> {path}")


if __name__ == "__main__":
    main()
