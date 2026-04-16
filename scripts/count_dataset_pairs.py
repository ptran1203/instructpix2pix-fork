from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_metadata_pairs(dataset_dir: Path) -> tuple[int, int, int]:
    prompt_dirs = 0
    metadata_pairs = 0
    complete_image_pairs = 0

    for prompt_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        prompt_dirs += 1
        metadata_path = prompt_dir / "metadata.jsonl"
        if not metadata_path.exists():
            continue

        with metadata_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                metadata_pairs += 1
                record = json.loads(line)
                seed = record["seed"]
                image_0 = prompt_dir / f"{seed}_0.jpg"
                image_1 = prompt_dir / f"{seed}_1.jpg"
                if image_0.exists() and image_1.exists():
                    complete_image_pairs += 1

    return prompt_dirs, metadata_pairs, complete_image_pairs


def count_prepared_pairs(dataset_dir: Path) -> int | None:
    seeds_path = dataset_dir / "seeds.json"
    if not seeds_path.exists():
        return None

    with seeds_path.open() as f:
        seeds = json.load(f)

    return sum(len(seed_list) for _, seed_list in seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count image pairs in an InstructPix2Pix dataset directory.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/generated_dataset",
        help="Path to dataset directory (default: data/generated_dataset)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    prompt_dirs, metadata_pairs, complete_image_pairs = count_metadata_pairs(dataset_dir)
    prepared_pairs = count_prepared_pairs(dataset_dir)

    print(f"Dataset: {dataset_dir}")
    print(f"Prompt directories: {prompt_dirs}")
    print(f"Pairs listed in metadata.jsonl: {metadata_pairs}")
    print(f"Complete image pairs found (*_0.jpg + *_1.jpg): {complete_image_pairs}")
    if prepared_pairs is not None:
        print(f"Pairs referenced by seeds.json: {prepared_pairs}")

    if metadata_pairs != complete_image_pairs:
        print("Warning: metadata pair count does not match complete image pair count.")
    if prepared_pairs is not None and prepared_pairs != complete_image_pairs:
        print("Warning: seeds.json pair count does not match complete image pair count.")


if __name__ == "__main__":
    main()
