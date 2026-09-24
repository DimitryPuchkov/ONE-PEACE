"""Phase 2: extend the GPT-2 BPE tokenizer with Russian merges (continuation training).

Reads the original tokenizer files (one_peace/utils/BPE), learns NEW BPE merge
rules on a Russian corpus while keeping every old merge and every old token id
fixed, and writes the extended tokenizer into a new folder (default
one_peace/utils/BPE_RU) consisting of encoder.json / vocab.bpe / dict.txt,
fully compatible with the existing loading code (just point --bpe-dir there).

Why appending is safe: the new merges are only allowed to produce pieces
containing at least one Cyrillic lead byte (UTF-8 bytes 0xD0-0xD4). Such
bytes occur exclusively inside Cyrillic text, so new merges can never fire
in English (or any other non-Cyrillic) text -> English tokenization is
provably unchanged; all old ids and their embeddings stay meaningful.

=============================================================================
CORPUS FORMAT (важно для написания конвертера)
=============================================================================
The corpus is a PLAIN UTF-8 TEXT FILE, ONE TEXT UNIT PER LINE
(предложение / подпись / короткий абзац):

    строка 1: Человек в красном шлеме на маленьком мопеде на грунтовой дороге.
    строка 2: Мужчина едет на мотоцикле по грунтовой дороге ...
    ...

Rules:
  * encoding: UTF-8, one text per line, lines separated by '\\n';
  * no ids, no quoting, no markup needed (tabs inside a line are tolerated);
  * raw case and punctuation are fine -- the script applies the same
    normalization as the training pipeline (lowercase, whitespace collapse,
    see normalize() below), so do NOT lowercase in the converter;
  * empty lines are skipped;
  * optional .gz compression is detected by the file extension.

Alternatively the script reads the project's translated TSV datasets directly:
    --tsv one_peace/dataset/mscoco/mscoco/train_translated.tsv --column caption
so for the current MSCOCO data no converter is needed at all.

A converter for any other source boils down to:
    with open('corpus.ru.txt', 'w', encoding='utf-8') as f:
        for text in my_texts:            # любые русские тексты
            f.write(text.replace('\\n', ' ').strip() + '\\n')

Usage (conda env one_peace), full run:
    python scripts/bpe_ru/phase2_extend_bpe.py \
        --corpus russian_corpus_download/ru_wiki.txt \
                russian_corpus_download/mscoco_captions.txt \
        --num-new-tokens 20000 --min-pair-freq 50 \
        --check-en-tsv one_peace/dataset/mscoco/mscoco/val_translated.tsv \
        --save-report scripts/bpe_ru/reports/phase2_extension.json
"""

import argparse
import gzip
import heapq
import importlib.util
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BPE_DIR = REPO_ROOT / "one_peace" / "utils" / "BPE"
DEFAULT_OUT_DIR = REPO_ROOT / "one_peace" / "utils" / "BPE_RU"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_baseline import normalize, read_tsv  # noqa: E402  (same normalization as the pipeline)

ENGLISH_CHECK_SENTENCES = [
    "hello world! It is me",
    "A man with a red helmet on a small moped on a dirt road.",
    "The quick brown fox jumps over the lazy dog.",
    "Two dogs play with a frisbee in the park at sunset.",
    "A group of people sitting around a dinner table, 2023.",
]


def load_encoder(bpe_dir: Path):
    encoder_json = bpe_dir / "encoder.json"
    vocab_bpe = bpe_dir / "vocab.bpe"
    for f in (encoder_json, vocab_bpe):
        if not f.is_file():
            sys.exit(f"error: {f} not found")
    try:
        from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "gpt2_bpe_utils",
            REPO_ROOT / "fairseq" / "fairseq" / "data" / "encoders" / "gpt2_bpe_utils.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_encoder = mod.get_encoder
    return get_encoder(str(encoder_json), str(vocab_bpe))


def iter_corpus_lines(args):
    """Yield raw text lines from the corpus (see format description above)."""
    if args.tsv:
        for ru, _en in read_tsv(args.tsv, args.column, args.column, 0):
            yield ru
    else:
        for corpus in args.corpus:
            if not corpus.is_file():
                sys.exit(f"error: corpus file {corpus} not found")
            opener = gzip.open if corpus.suffix == ".gz" else open
            with opener(corpus, "rt", encoding="utf-8") as f:
                for line in f:
                    yield line


def collect_word_freq(encoder, lines, max_lines):
    """Pretokenize + byte-map the corpus into a word->frequency table.

    'word' here is a regex pretoken (e.g. ' мир' -> 'Ġмир' in the byte-mapped
    space), exactly what Encoder.encode() would feed into bpe() at runtime.
    """
    word_freq = Counter()
    n_lines = 0
    for line in lines:
        for token in encoder.re.findall(encoder.pat, normalize(line)):
            mapped = "".join(encoder.byte_encoder[b] for b in token.encode("utf-8"))
            if mapped:
                word_freq[mapped] += 1
        n_lines += 1
        if max_lines and n_lines >= max_lines:
            break
    return word_freq, n_lines


def merge_word(word, first, second):
    """Apply one merge to a symbol tuple.

    Replicates the exact loop of Encoder.bpe() in gpt2_bpe_utils.py (lines
    86-103): left-to-right scan, non-overlapping occurrences -- so the
    trainer's segmentations match what the runtime tokenizer will produce.
    """
    new_word = []
    i = 0
    while i < len(word):
        try:
            j = word.index(first, i)
            new_word.extend(word[i:j])
            i = j
        except ValueError:
            new_word.extend(word[i:])
            break
        if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def cyrillic_lead_chars(encoder):
    """Mapped chars of the UTF-8 lead bytes 0xD0-0xD4 (Cyrillic U+0400-U+052F).

    Byte 0xD0..0xD4 occurs in UTF-8 text ONLY as the lead byte of a Cyrillic
    codepoint, so a piece containing one of these chars can only ever match
    inside Cyrillic text. NB: checking plain ASCII is NOT enough -- pieces
    starting with a space always contain the non-ASCII char 'Ġ' (U+0120), so
    an ASCII-only filter would let e.g. 'Ġwii' (latin brand left in the RU
    corpus by the translator) slip through and change English tokenization.
    """
    return {encoder.byte_encoder[b] for b in range(0xD0, 0xD5)}


def train_merges(encoder, word_freq, num_new_tokens, min_pair_freq, log_every=2000):
    """Continuation BPE training: old merges fixed, learn new ones.

    Returns (new_merges, final_segmentations, stats). Classic incremental
    algorithm: count pairs once, then after each selected merge update counts
    only for the affected words; a lazy max-heap yields the best pair.
    """
    t0 = time.time()
    old_ranks = encoder.bpe_ranks
    cyr_chars = cyrillic_lead_chars(encoder)

    def is_safe_piece(piece):
        # only merges touching Cyrillic lead bytes; such pairs cannot fire
        # in any non-Cyrillic text (incl. English with typographic quotes)
        return any(c in cyr_chars for c in piece)

    # initial segmentation: existing bpe() exhausts all old merges
    words, freqs, segs = [], [], []
    for w, freq in word_freq.items():
        words.append(w)
        freqs.append(freq)
        segs.append(tuple(encoder.bpe(w).split(" ")))

    def pairs_of(seq):
        return zip(seq, seq[1:])

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for i, seq in enumerate(segs):
        f = freqs[i]
        for a, b in pairs_of(seq):
            if not is_safe_piece(a + b):
                continue  # never learn non-Cyrillic merges (English safety)
            pair_counts[(a, b)] += f
            pair_to_words[(a, b)].add(i)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    new_merges = []
    merged_pairs = set()
    skipped_ascii = 0
    while len(new_merges) < num_new_tokens and heap:
        negc, pair = heapq.heappop(heap)
        c = pair_counts.get(pair, 0)
        if c != -negc:
            continue  # stale heap entry
        if c < min_pair_freq:
            break  # everything left is below the frequency floor
        first, second = pair
        piece = first + second
        if not is_safe_piece(piece) or piece in encoder.encoder or pair in merged_pairs:
            # belt-and-suspenders: same guards as during counting
            del pair_counts[pair]
            skipped_ascii += 1
            continue
        touched = set()
        for i in list(pair_to_words[pair]):
            old_seq = segs[i]
            new_seq = merge_word(old_seq, first, second)
            if new_seq == old_seq:
                continue  # occurrences not adjacent anymore (stale index)
            f = freqs[i]
            for p in pairs_of(old_seq):
                pair_counts[p] -= f
                touched.add(p)
            for p in pairs_of(new_seq):
                pair_counts[p] += f
                pair_to_words[p].add(i)
                touched.add(p)
            segs[i] = new_seq
        # heap is lazy: push a fresh entry for every changed pair; stale
        # entries are skipped on pop by comparing with pair_counts
        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
        new_merges.append(pair)
        merged_pairs.add(pair)
        if len(new_merges) % log_every == 0:
            best = max(pair_counts.values()) if pair_counts else 0
            print(
                f"  merges: {len(new_merges)}/{num_new_tokens}, "
                f"best remaining pair freq: {best}, elapsed {time.time()-t0:.0f}s",
                flush=True,
            )

    stats = {
        "unique_words": len(words),
        "new_merges": len(new_merges),
        "skipped_ascii_or_known": skipped_ascii,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return new_merges, list(zip(words, freqs, segs)), stats


def write_tokenizer(out_dir: Path, bpe_dir: Path, encoder, new_merges, word_segs):
    """Materialize BPE_RU: encoder.json, vocab.bpe, dict.txt (+ __init__.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_id = max(encoder.encoder.values()) + 1  # 50257 for the stock GPT-2 vocab

    # encoder.json: old entries + one entry per new merge result
    new_encoder = dict(encoder.encoder)
    piece_counts = Counter()
    for _w, freq, seq in word_segs:
        for sym in seq:
            piece_counts[sym] += freq
    for idx, (a, b) in enumerate(new_merges):
        new_encoder[a + b] = base_id + idx
    with open(out_dir / "encoder.json", "w", encoding="utf-8") as f:
        # ensure_ascii keeps the file style identical to the original GPT-2 one
        json.dump(new_encoder, f, ensure_ascii=True)

    # vocab.bpe: header + old merges + new merges appended AFTER all old ones,
    # so their ranks (50001+) never outrank an old merge
    old_vocab_text = (bpe_dir / "vocab.bpe").read_text(encoding="utf-8")
    old_body = old_vocab_text.rstrip("\n")
    new_lines = [old_body] + [f"{a} {b}" for a, b in new_merges]
    (out_dir / "vocab.bpe").write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # dict.txt: old lines + one line per new token ("gpt2_id count"),
    # final model id = dict.txt line number + 3 (4 fairseq specials - 1)
    old_dict_text = (bpe_dir / "dict.txt").read_text(encoding="utf-8")
    old_dict_text = old_dict_text if old_dict_text.endswith("\n") else old_dict_text + "\n"
    dict_lines = [old_dict_text]
    for idx, (a, b) in enumerate(new_merges):
        piece = a + b
        dict_lines.append(f"{base_id + idx} {piece_counts.get(piece, 0)}\n")
    (out_dir / "dict.txt").write_text("".join(dict_lines), encoding="utf-8")

    src_init = bpe_dir / "__init__.py"
    if src_init.is_file():
        shutil.copy(src_init, out_dir / "__init__.py")
    else:
        (out_dir / "__init__.py").touch()

    return base_id, piece_counts


def verify(out_dir: Path, old_encoder, new_merges, corpus_lines, en_texts, max_check=20000):
    """Built-in phase-4 style checks on the freshly written tokenizer."""
    new_encoder = load_encoder(out_dir)
    problems = []

    # 1. English invariance (the core safety property)
    en_texts = (ENGLISH_CHECK_SENTENCES + en_texts)[:max_check]
    mismatched = 0
    for text in en_texts:
        if new_encoder.encode(text) != old_encoder.encode(text):
            mismatched += 1
    if mismatched:
        problems.append(f"ENGLISH TOKENIZATION CHANGED on {mismatched}/{len(en_texts)} texts")

    # 2. Russian fertility on a corpus sample, old vs new
    sample = corpus_lines[:max_check]
    old_tokens = new_tokens = words_n = 0
    for line in sample:
        line = normalize(line)
        words_n += len(line.split())
        old_tokens += len(old_encoder.encode(line))
        new_tokens += len(new_encoder.encode(line))
    fertility_old = old_tokens / max(words_n, 1)
    fertility_new = new_tokens / max(words_n, 1)

    # 3. round-trip + consistency
    for line in sample[:2000]:
        line = normalize(line)
        if line and new_encoder.decode(new_encoder.encode(line)) != line:
            problems.append(f"ROUNDTRIP FAILED on {line[:60]!r}")
            break
    n_dict = len((out_dir / "dict.txt").read_text(encoding="utf-8").splitlines())
    n_enc = len(json.loads((out_dir / "encoder.json").read_text(encoding="utf-8")))
    if n_enc != 50257 + len(new_merges):
        problems.append(f"encoder.json size {n_enc} != {50257 + len(new_merges)}")
    if n_dict != 50260 + len(new_merges):
        problems.append(f"dict.txt size {n_dict} != {50260 + len(new_merges)}")

    return {
        "english_texts_checked": len(en_texts),
        "english_mismatches": mismatched,
        "fertility_old": round(fertility_old, 2),
        "fertility_new": round(fertility_new, 2),
        "sample_words": words_n,
        "dict_lines": n_dict,
        "encoder_entries": n_enc,
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", type=Path, nargs="+",
                     help="corpus: plain UTF-8 text files, one text per line each (.gz ok); "
                          "several files are concatenated (e.g. wiki + captions)")
    src.add_argument("--tsv", type=Path, help="alternatively: project TSV dataset")
    parser.add_argument("--column", default="caption", help="text column when --tsv is used")
    parser.add_argument("--bpe-dir", type=Path, default=DEFAULT_BPE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-new-tokens", type=int, default=20000)
    parser.add_argument("--min-pair-freq", type=int, default=50,
                        help="stop when the best remaining pair is rarer than this")
    parser.add_argument("--max-corpus-lines", type=int, default=0, help="0 = all")
    parser.add_argument("--check-en-tsv", type=Path, default=None,
                        help="optional TSV (en_caption column) for a large English invariance check")
    parser.add_argument("--save-report", type=Path, default=None)
    args = parser.parse_args()

    print(f"loading tokenizer from {args.bpe_dir}")
    encoder = load_encoder(args.bpe_dir)

    print("collecting word frequencies from corpus ...")
    lines = list(iter_corpus_lines(args))
    word_freq, n_lines = collect_word_freq(encoder, lines, args.max_corpus_lines)
    print(f"  {n_lines} lines -> {sum(word_freq.values())} pretokens, "
          f"{len(word_freq)} unique words")

    print(f"training up to {args.num_new_tokens} new merges "
          f"(min pair freq {args.min_pair_freq}) ...")
    new_merges, word_segs, train_stats = train_merges(
        encoder, word_freq, args.num_new_tokens, args.min_pair_freq
    )
    print(f"  done: {train_stats}")

    print(f"writing extended tokenizer to {args.out_dir}")
    base_id, piece_counts = write_tokenizer(args.out_dir, args.bpe_dir, encoder, new_merges, word_segs)

    en_texts = []
    if args.check_en_tsv:
        en_texts = [en for _ru, en in read_tsv(args.check_en_tsv, "en_caption", "en_caption", 0)]
    print("verifying ...")
    check = verify(args.out_dir, encoder, new_merges, lines, en_texts)

    new_encoder = load_encoder(args.out_dir)
    top = [
        {"id": base_id + i, "piece": a + b, "readable": new_encoder.decode([base_id + i]),
         "count": piece_counts.get(a + b, 0)}
        for i, (a, b) in enumerate(new_merges[:20])
    ]

    print("\n== Result ==")
    print(f"  new tokens: {len(new_merges)} (ids {base_id}..{base_id + len(new_merges) - 1})")
    print(f"  RU fertility on sample: {check['fertility_old']} -> {check['fertility_new']} tok/word")
    print(f"  English invariance: {check['english_mismatches']} mismatches "
          f"on {check['english_texts_checked']} texts")
    print(f"  dict.txt lines: {check['dict_lines']}, encoder.json entries: {check['encoder_entries']}")
    print("\n  first 20 new tokens:")
    for t in top:
        print(f"    id={t['id']:<6} count={t['count']:<8} piece={t['piece']!r} -> {t['readable']!r}")
    if check["problems"]:
        print("\nPROBLEMS DETECTED:", *check["problems"], sep="\n  ")
        sys.exit(1)
    print("\nall checks passed")

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps({
            "args": {k: str(v) for k, v in vars(args).items()},
            "train_stats": train_stats,
            "verify": check,
            "top_new_tokens": top,
        }, ensure_ascii=False, indent=2))
        print(f"report saved to {args.save_report}")


if __name__ == "__main__":
    main()
