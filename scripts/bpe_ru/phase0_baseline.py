"""Phase 0: baseline tokenization stats for Russian vs English text.

Measures how the current GPT-2 BPE tokenizer (one_peace/utils/BPE) handles
Russian captions compared to the original English ones, using the same
normalization as the training pipeline (one_peace/data/base_dataset.py,
process_text) so the numbers reflect runtime behaviour.

Usage (conda env one_peace):
    python scripts/bpe_ru/phase0_baseline.py \
        --tsv one_peace/dataset/mscoco/mscoco/val_translated.tsv \
        --save-report scripts/bpe_ru/reports/phase0_baseline.json

Later, after building the extended tokenizer, run the same script with
--bpe-dir one_peace/utils/BPE_RU to compare against this baseline.
"""

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BPE_DIR = REPO_ROOT / "one_peace" / "utils" / "BPE"
DEFAULT_TSV = REPO_ROOT / "one_peace" / "dataset" / "mscoco" / "mscoco" / "val_translated.tsv"
# max_src_length from one_peace/tasks/base_task.py, captions longer than this are truncated
MAX_SRC_LENGTH = 70


def load_encoder(bpe_dir: Path):
    encoder_json = bpe_dir / "encoder.json"
    vocab_bpe = bpe_dir / "vocab.bpe"
    for f in (encoder_json, vocab_bpe):
        if not f.is_file():
            sys.exit(f"error: {f} not found")

    try:
        from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
    except ImportError:
        # env without installed fairseq: load the module file directly
        # (it only depends on json + regex)
        spec = importlib.util.spec_from_file_location(
            "gpt2_bpe_utils",
            REPO_ROOT / "fairseq" / "fairseq" / "data" / "encoders" / "gpt2_bpe_utils.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_encoder = mod.get_encoder

    return get_encoder(str(encoder_json), str(vocab_bpe))


def normalize(text: str) -> str:
    """Same normalization as BaseDataset.process_text (base_dataset.py:99).

    Note the .lower(): the training pipeline always lowercases, so the
    baseline must be measured on lowercased text as well.
    """
    text = text.lower().lstrip(",.!?*#:;~")
    text = re.sub(r"\s{2,}|\t", " ", text)
    text = text.rstrip("\n")
    return text.strip(" ")


def read_tsv(path: Path, ru_col: str, en_col: str, limit: int):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if ru_col not in reader.fieldnames or en_col not in reader.fieldnames:
            sys.exit(
                f"error: columns {ru_col!r}/{en_col!r} not found in {path}, "
                f"available: {reader.fieldnames}"
            )
        rows = []
        for row in reader:
            ru, en = (row.get(ru_col) or "").strip(), (row.get(en_col) or "").strip()
            if ru and en:
                rows.append((ru, en))
            if limit and len(rows) >= limit:
                break
    return rows


def token_len_class(piece: str) -> str:
    # piece is in the byte-to-unicode mapped space; a merged multi-char piece
    # has length > 1 there
    return "single_byte" if len(piece) == 1 else "merged"


def measure(encoder, texts):
    stats = {
        "texts": 0,
        "words": 0,
        "tokens": 0,
        "tokens_per_text": [],
        "piece_classes": Counter(),
        "piece_chars": 0,  # total chars of readable piece text
        "token_freq": Counter(),
        "roundtrip_errors": 0,
        "over_max_len": 0,
        "example": None,
    }
    rev = {v: k for k, v in encoder.encoder.items()}

    for i, text in enumerate(texts):
        text = normalize(text)
        if not text:
            continue
        ids = encoder.encode(text)
        pieces = [rev[t] for t in ids]

        stats["texts"] += 1
        stats["words"] += len(text.split())
        stats["tokens"] += len(ids)
        stats["tokens_per_text"].append(len(ids))
        stats["token_freq"].update(ids)
        for piece in pieces:
            stats["piece_classes"][token_len_class(piece)] += 1
            stats["piece_chars"] += len(piece)
        decoded = encoder.decode(ids)
        if decoded != text:
            stats["roundtrip_errors"] += 1
        if len(ids) > MAX_SRC_LENGTH:
            stats["over_max_len"] += 1
        if stats["example"] is None:
            stats["example"] = (text, pieces)

    n = max(stats["texts"], 1)
    sorted_lens = sorted(stats["tokens_per_text"])
    p95 = sorted_lens[int(len(sorted_lens) * 0.95)] if sorted_lens else 0

    report = {
        "texts": stats["texts"],
        "words": stats["words"],
        "tokens": stats["tokens"],
        "tokens_per_word": round(stats["tokens"] / max(stats["words"], 1), 2),
        "tokens_per_text_avg": round(stats["tokens"] / n, 1),
        "tokens_per_text_p95": p95,
        f"texts_over_{MAX_SRC_LENGTH}_tokens": stats["over_max_len"],
        f"share_over_{MAX_SRC_LENGTH}_tokens": round(stats["over_max_len"] / n, 3),
        "single_byte_piece_share": round(
            stats["piece_classes"]["single_byte"] / max(stats["tokens"], 1), 3
        ),
        "unique_tokens_used": len(stats["token_freq"]),
        "avg_piece_chars": round(stats["piece_chars"] / max(stats["tokens"], 1), 2),
        "roundtrip_errors": stats["roundtrip_errors"],
    }
    # most frequent tokens, rendered readable
    top = []
    for tid, cnt in stats["token_freq"].most_common(20):
        piece = rev[tid]
        readable = encoder.decode([tid])
        top.append({"gpt2_id": tid, "piece": piece, "readable": readable, "count": cnt})
    return report, top, stats["example"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--ru-column", default="caption")
    parser.add_argument("--en-column", default="en_caption")
    parser.add_argument("--bpe-dir", type=Path, default=DEFAULT_BPE_DIR)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--save-report", type=Path, default=None)
    args = parser.parse_args()

    encoder = load_encoder(args.bpe_dir)
    rows = read_tsv(args.tsv, args.ru_column, args.en_column, args.limit)
    print(f"loaded {len(rows)} rows with both RU and EN text from {args.tsv}")
    print(f"tokenizer: {args.bpe_dir}\n")

    ru_report, ru_top, ru_example = measure(encoder, (ru for ru, _ in rows))
    en_report, en_top, en_example = measure(encoder, (en for _, en in rows))

    def fmt(label, r):
        return (
            f"  {label}: fertility={r['tokens_per_word']} tok/word | "
            f"{r['tokens_per_text_avg']} tok/text (p95={r['tokens_per_text_p95']}) | "
            f"single-byte pieces: {r['single_byte_piece_share']*100:.1f}% | "
            f"avg piece {r['avg_piece_chars']} chars | "
            f">{MAX_SRC_LENGTH} tokens: {r[f'texts_over_{MAX_SRC_LENGTH}_tokens']} ({r[f'share_over_{MAX_SRC_LENGTH}_tokens']*100:.1f}%)"
        )

    print("== Fertility baseline ==")
    print(fmt("RU", ru_report))
    print(fmt("EN", en_report))
    print(f"  ratio RU/EN fertility: {round(ru_report['tokens_per_word'] / max(en_report['tokens_per_word'], 1e-9), 2)}x")
    print(f"  unique tokens: RU={ru_report['unique_tokens_used']}, EN={en_report['unique_tokens_used']} (vocab={len(encoder.encoder)})")
    print(f"  roundtrip decode errors: RU={ru_report['roundtrip_errors']}, EN={en_report['roundtrip_errors']}")

    print("\n== Example (RU) ==")
    text, pieces = ru_example
    print(f"  text:   {text!r}")
    print(f"  pieces: {[encoder.decode([encoder.encoder[p]]) for p in pieces][:24]}")

    print("\n== Top-20 RU tokens (mostly byte fragments => no real Russian merges) ==")
    for t in ru_top:
        print(f"  {t['count']:>7}  id={t['gpt2_id']:<6} piece={t['piece']!r}  -> {t['readable']!r}")

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tsv": str(args.tsv),
            "bpe_dir": str(args.bpe_dir),
            "rows": len(rows),
            "ru": ru_report,
            "en": en_report,
            "ru_top_tokens": ru_top,
        }
        args.save_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\nreport saved to {args.save_report}")


if __name__ == "__main__":
    main()
