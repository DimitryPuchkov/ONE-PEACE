"""Phase 5: checkpoint surgery for the extended RU vocabulary.

Resizes the vocabulary-dependent embedding matrix inside a fairseq checkpoint
from the original GPT-2 dictionary (one_peace/utils/BPE, 50264 rows incl. 4
fairseq specials) to the extended one (one_peace/utils/BPE_RU, len(dict.txt)
+ 4 rows), so ONE-PEACE pretraining can continue with Russian text
(RuAdapt-style adaptation):

  * every tensor whose first dim equals the old vocab size and whose name
    contains 'embed_tokens' is resized (in practice exactly one:
    'encoder_wrapper.text_adapter.embed_tokens.weight'; the decoder wrapper is
    built without a dictionary and the mask heads live in feature space);
  * old rows are copied bit-for-bit -- all original token embeddings,
    including padding_idx=1 zeros, keep their meaning;
  * new rows are initialized by 'byte_mean': the new token's piece string is
    decomposed into single-byte-mapped chars (each of them is an original
    GPT-2 token with a trained embedding) and their mean is used. This starts
    new tokens from the composition of their parts instead of noise
    (--init-mode random for N(0, std_of_old_rows) instead);
  * optimizer/extra state is dropped by default (--keep-optimizer-state to
    keep) -- Adam moments have the old parameter shapes and cannot be reused
    after a vocab extension;
  * if the checkpoint embeds a task config with bpe_dir, it is pointed to the
    new tokenizer dir.

Run on the machine where the checkpoint lives (e.g. the cluster), env
one_peace:
    python scripts/bpe_ru/phase5_checkpoint_surgery.py \
        --checkpoint checkpoints/pretrain/one_peace_video_60_10epoch/checkpoint3.pt \
        --out      checkpoints/pretrain/checkpoint3_bpe_ru.pt

Then continue pretraining with the usual script, only changing:
    task.bpe_dir=<repo>/one_peace/utils/BPE_RU
    checkpoint.restore_file=<...>/checkpoint3_bpe_ru.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_BPE = REPO_ROOT / "one_peace" / "utils" / "BPE"
DEFAULT_NEW_BPE = REPO_ROOT / "one_peace" / "utils" / "BPE_RU"

# fairseq Dictionary adds 4 specials (<s>, <pad>, </s>, <unk>) before dict.txt
N_SPECIAL = 4


def load_dict_symbols(bpe_dir: Path):
    """Token strings from dict.txt, ordered by final model id."""
    lines = (bpe_dir / "dict.txt").read_text(encoding="utf-8").splitlines()
    return [line.split(" ")[0] for line in lines]


def load_encoder_json(bpe_dir: Path):
    return json.loads((bpe_dir / "encoder.json").read_text(encoding="utf-8"))


def build_byte_piece_rows(old_symbols, old_encoder):
    """fairseq row id for every single-byte-mapped char token.

    Each char of a piece string is itself a GPT-2 base token (ids 0..255 in
    encoder.json); its fairseq id = position of str(gpt2_id) in dict.txt + 4.
    """
    line_of_token = {tok: i for i, tok in enumerate(old_symbols)}
    char_rows = {}
    for char, gpt2_id in old_encoder.items():
        if len(char) == 1:
            row = line_of_token.get(str(gpt2_id))
            if row is not None:
                char_rows[char] = row + N_SPECIAL
    return char_rows


def torch_load(path):
    # fairseq checkpoints embed cfg/args objects -> weights_only=False is
    # required (trusted files only: own checkpoints / official releases)
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <checkpoint>_bpe_ru.pt next to it")
    parser.add_argument("--old-bpe-dir", type=Path, default=DEFAULT_OLD_BPE)
    parser.add_argument("--new-bpe-dir", type=Path, default=DEFAULT_NEW_BPE)
    parser.add_argument("--init-mode", choices=["byte_mean", "random"], default="byte_mean")
    parser.add_argument("--keep-optimizer-state", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        sys.exit(f"error: checkpoint {args.checkpoint} not found")
    args.out = args.out or args.checkpoint.with_name(
        args.checkpoint.stem + "_bpe_ru" + args.checkpoint.suffix
    )

    old_symbols = load_dict_symbols(args.old_bpe_dir)
    new_symbols = load_dict_symbols(args.new_bpe_dir)
    old_vocab = len(old_symbols) + N_SPECIAL      # 50264
    new_vocab = len(new_symbols) + N_SPECIAL      # 50264 + 19250
    assert old_symbols == new_symbols[: len(old_symbols)], (
        "dict.txt of the new tokenizer must start with the old lines verbatim"
    )
    print(f"old vocab {old_vocab} -> new vocab {new_vocab} "
          f"(+{new_vocab - old_vocab} tokens)")

    old_encoder = load_encoder_json(args.old_bpe_dir)
    new_encoder = load_encoder_json(args.new_bpe_dir)
    inv_new = {v: k for k, v in new_encoder.items()}
    char_rows = build_byte_piece_rows(old_symbols, old_encoder)

    print(f"loading checkpoint {args.checkpoint} (cpu) ...")
    state = torch_load(args.checkpoint)

    model_sd = state.get("model", state)
    resized = []
    for key, tensor in list(model_sd.items()):
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            continue
        if "embed_tokens" in key and tensor.shape[0] == old_vocab:
            new_rows = init_new_rows(
                tensor, old_vocab, new_vocab, new_symbols, inv_new, char_rows,
                args.init_mode, args.seed,
            )
            model_sd[key] = torch.cat([tensor, new_rows], dim=0)
            resized.append((key, tuple(tensor.shape), tuple(model_sd[key].shape)))
    if not resized:
        avail = {k: tuple(v.shape) for k, v in model_sd.items()
                 if torch.is_tensor(v) and v.ndim == 2}
        sys.exit(
            f"error: no '*embed_tokens*' tensor of shape[0]=={old_vocab} found. "
            f"2D tensors present: {avail}"
        )
    for key, old_shape, new_shape in resized:
        print(f"  resized {key}: {old_shape} -> {new_shape}")

    # any other old-vocab-shaped tensor would silently misalign -> hard error
    leftovers = [k for k, v in model_sd.items()
                 if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == old_vocab]
    if leftovers:
        sys.exit(f"error: other tensors with first dim {old_vocab} remain: {leftovers}")

    if not args.keep_optimizer_state:
        for meta_key in ("last_optimizer_state", "optimizer_history", "extra_state"):
            if state.pop(meta_key, None) is not None:
                print(f"  dropped checkpoint field '{meta_key}' "
                      "(incompatible shapes after vocab extension)")

    # point the embedded task config at the new tokenizer dir (dict or OmegaConf)
    if patch_cfg_bpe_dir(state.get("cfg"), str(args.new_bpe_dir)):
        print(f"  checkpoint cfg.task.bpe_dir -> {args.new_bpe_dir}")
    ckpt_args = state.get("args")
    if ckpt_args is not None and getattr(ckpt_args, "bpe_dir", None) is not None:
        ckpt_args.bpe_dir = str(args.new_bpe_dir)
        print(f"  checkpoint args.bpe_dir -> {ckpt_args.bpe_dir}")

    torch.save(state, args.out)
    print(f"saved {args.out}")

    verify(args.out, resized, old_vocab)


def patch_cfg_bpe_dir(cfg, new_dir):
    """Point the embedded task config at the new tokenizer dir.

    The checkpoint may store cfg as a plain dict or as OmegaConf; handle both.
    """
    if cfg is None:
        return False
    task = cfg.get("task") if isinstance(cfg, dict) else getattr(cfg, "task", None)
    if task is None:
        return False
    if isinstance(task, dict):
        if "bpe_dir" in task:
            task["bpe_dir"] = new_dir
            return True
    else:
        try:
            if "bpe_dir" in task:
                task.bpe_dir = new_dir
                return True
        except Exception:
            pass
    return False


def init_new_rows(old_embed, old_vocab, new_vocab, new_symbols, inv_new,
                  char_rows, mode, seed):
    n_new, dim = new_vocab - old_vocab, old_embed.shape[1]
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.empty(n_new, dim).normal_(
            0, old_embed.float().std().item(), generator=g
        ).to(old_embed.dtype)

    rows = torch.zeros(n_new, dim, dtype=old_embed.dtype)
    fallback = 0
    for j in range(n_new):
        final_id = old_vocab + j
        token = new_symbols[final_id - N_SPECIAL]      # e.g. "50257"
        piece = inv_new[int(token)]                    # e.g. 'ĠÑģ' (= ' с')
        part_rows = [char_rows[c] for c in piece if c in char_rows]
        if part_rows:
            rows[j] = old_embed[part_rows].mean(dim=0)
        else:
            fallback += 1
    if fallback:
        print(f"  warning: {fallback} new tokens had no byte pieces, left zero")
    return rows


def verify(out_path, resized, old_vocab):
    state = torch_load(out_path)
    model_sd = state.get("model", state)
    for key, old_shape, new_shape in resized:
        t = model_sd[key]
        assert tuple(t.shape) == new_shape, f"{key}: shape {t.shape} != {new_shape}"
    key = resized[0][0]
    t_new = model_sd[key]
    # prefix rows must be identical to what was there before the resize
    # (spot-check via re-loading is impossible w/o the original, but shapes
    # and stats give a quick sanity signal)
    print(f"verify {out_path.name}: {key} {tuple(t_new.shape)}, "
          f"prefix norm {t_new[:old_vocab].float().norm():.1f}, "
          f"new-rows norm {t_new[old_vocab:].float().norm():.1f}")


if __name__ == "__main__":
    main()
