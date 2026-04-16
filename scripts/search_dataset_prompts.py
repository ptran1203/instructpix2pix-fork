from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator


def iter_prompt_files(dataset_dir: Path) -> Iterator[Path]:
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir():
            prompt_file = child / "prompt.json"
            if prompt_file.exists():
                yield prompt_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Search prompt.json files in an InstructPix2Pix dataset.")
    parser.add_argument("query", help="Word or phrase to search for, e.g. 'Leonardo DiCaprio'")
    parser.add_argument(
        "--dataset",
        default="data/generated_dataset",
        help="Dataset directory containing per-sample folders (default: data/generated_dataset)",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive matching (default: case-insensitive)",
    )
    parser.add_argument(
        "--regex",
        action="store_true",
        help="Treat query as a regular expression",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of matches to print (default: 50)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    flags = 0 if args.case_sensitive else re.IGNORECASE
    pattern = re.compile(args.query if args.regex else re.escape(args.query), flags)

    total_prompt_files = 0
    total_matches = 0

    for prompt_file in iter_prompt_files(dataset_dir):
        total_prompt_files += 1
        with prompt_file.open() as f:
            record = json.load(f)

        field_hits: list[str] = []
        for key, value in record.items():
            if isinstance(value, str) and pattern.search(value):
                field_hits.append(f"{key}: {value}")

        if field_hits:
            total_matches += 1
            print(f"[{prompt_file.parent.name}] {prompt_file}")
            for hit in field_hits:
                print(f"  - {hit}")
            print()
            if total_matches >= args.limit:
                break

    print(f"Scanned prompt files: {total_prompt_files}")
    print(f"Matching samples: {total_matches}")


if __name__ == "__main__":
    main()
