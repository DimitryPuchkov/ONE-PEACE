"""Convert the downloaded HuggingFace Russian Wikipedia dataset to the plain
corpus format accepted by phase2_extend_bpe.py.

Input: a dataset saved with `datasets.save_to_disk` (folder contains
dataset_dict.json and <split>/data-*.arrow). The .arrow files are read
directly via pyarrow (IPC stream), so the heavy `datasets` package is NOT
required in the training env.

Output format (see phase2_extend_bpe.py docstring): plain UTF-8 text, one
text unit per line, raw case/punctuation kept (the trainer normalizes itself).

Cleaning rules:
  * newlines/tabs inside a text are replaced with spaces (one line = one text);
  * repeated whitespace collapsed;
  * empty and whitespace-only texts are dropped;
  * texts with no Cyrillic character at all are dropped (--keep-all to
    disable): they cannot produce any Russian merge and only slow the trainer
    down. MIXED texts (Russian + Latin/digits/symbols) are always kept --
    non-Cyrillic fragments are harmless, the trainer only learns merges that
    contain Cyrillic UTF-8 lead bytes;
  * exact duplicate lines are dropped.

Usage (conda env one_peace):
    python scripts/bpe_ru/convert_hf_wiki.py \
        --dataset-dir russian_corpus_download/russian_corpus \
        --out russian_corpus_download/ru_wiki.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow as pa

WHITESPACE_RE = re.compile(r"\s+")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def read_arrow_table(arrow_path: Path) -> pa.Table:
    """Read a HF save_to_disk arrow shard (Arrow IPC stream, no footer)."""
    src = pa.memory_map(str(arrow_path))  # keep the reference: lazy streams
    try:
        return pa.ipc.open_stream(src).read_all()
    except pa.ArrowInvalid:
        src = pa.memory_map(str(arrow_path))
        return pa.ipc.open_file(src).read_all()


def iter_dataset_texts(dataset_dir: Path):
    dict_path = dataset_dir / "dataset_dict.json"
    if not dict_path.is_file():
        sys.exit(f"error: {dict_path} not found (is it a save_to_disk folder?)")
    splits = json.loads(dict_path.read_text())["splits"]
    for split in splits:
        shards = sorted((dataset_dir / split).glob("data-*.arrow"))
        if not shards:
            sys.exit(f"error: no data-*.arrow in {dataset_dir / split}")
        for shard in shards:
            table = read_arrow_table(shard)
            if "text" not in table.column_names:
                sys.exit(
                    f"error: expected a 'text' column, got {table.column_names} "
                    f"in {shard}"
                )
            yield from table.column("text").to_pylist()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="folder created by datasets.save_to_disk()")
    parser.add_argument("--out", type=Path, required=True,
                        help="output .txt file (one text per line)")
    parser.add_argument("--keep-all", action="store_true",
                        help="keep texts without any Cyrillic (default: drop)")
    args = parser.parse_args()

    stats = {"total": 0, "kept": 0, "dropped_empty": 0, "dropped_no_cyrillic": 0,
             "dropped_duplicates": 0}
    seen = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for text in iter_dataset_texts(args.dataset_dir):
            stats["total"] += 1
            if text is None:
                stats["dropped_empty"] += 1
                continue
            line = WHITESPACE_RE.sub(" ", text).strip()
            if not line:
                stats["dropped_empty"] += 1
                continue
            if not args.keep_all and not CYRILLIC_RE.search(line):
                stats["dropped_no_cyrillic"] += 1
                continue
            if line in seen:
                stats["dropped_duplicates"] += 1
                continue
            seen.add(line)
            f.write(line + "\n")
            stats["kept"] += 1

    print(f"written {stats['kept']} lines to {args.out}")
    print(f"  dropped: {stats['dropped_empty']} empty, "
          f"{stats['dropped_no_cyrillic']} without Cyrillic, "
          f"{stats['dropped_duplicates']} duplicates (of {stats['total']} total)")


if __name__ == "__main__":
    main()
