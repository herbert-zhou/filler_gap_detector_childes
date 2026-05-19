from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from transformers import GPT2TokenizerFast


@dataclass
class PairResult:
    dataset: str
    split: str
    ablated_sentences: int
    control_sentences: int
    ablated_tokens: int
    control_tokens: int

    @property
    def sentence_match(self) -> bool:
        return self.ablated_sentences == self.control_sentences

    @property
    def token_match(self) -> bool:
        return self.ablated_tokens == self.control_tokens

    @property
    def ok(self) -> bool:
        return self.sentence_match and self.token_match


def get_tokenizer(tokenizer_path: Path) -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    return tokenizer


def count_sentences_and_tokens(file_path: Path, tokenizer: GPT2TokenizerFast) -> tuple[int, int]:
    sentence_count = 0
    token_count = 0

    with file_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            sentence = line.strip()
            if not sentence:
                continue
            sentence_count += 1
            token_count += len(tokenizer(sentence, add_special_tokens=False)["input_ids"])

    return sentence_count, token_count


def iter_dataset_pairs(filtered_root: Path) -> Iterable[tuple[str, Path, Path]]:
    for control_dir in sorted(filtered_root.iterdir()):
        if not control_dir.is_dir() or not control_dir.name.endswith("_control"):
            continue

        dataset = control_dir.name[: -len("_control")]
        ablated_dir = filtered_root / dataset
        if ablated_dir.exists() and ablated_dir.is_dir():
            yield dataset, ablated_dir, control_dir


def verify(
    filtered_root: Path,
    tokenizer: GPT2TokenizerFast,
    splits: list[str],
) -> list[PairResult]:
    results: list[PairResult] = []

    for dataset, ablated_dir, control_dir in iter_dataset_pairs(filtered_root):
        for split in splits:
            ablated_file = ablated_dir / split / f"{split}.txt"
            control_file = control_dir / split / f"{split}.txt"

            if not ablated_file.exists() or not control_file.exists():
                continue

            ablated_sentences, ablated_tokens = count_sentences_and_tokens(ablated_file, tokenizer)
            control_sentences, control_tokens = count_sentences_and_tokens(control_file, tokenizer)

            results.append(
                PairResult(
                    dataset=dataset,
                    split=split,
                    ablated_sentences=ablated_sentences,
                    control_sentences=control_sentences,
                    ablated_tokens=ablated_tokens,
                    control_tokens=control_tokens,
                )
            )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that control datasets match ablated datasets in sentence and token counts."
    )
    parser.add_argument(
        "--filtered-root",
        type=Path,
        default=Path("data/filtered/par"),
        help="Root directory containing ablated and *_control datasets.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("models/gpt-clean-16000.json"),
        help="Path to tokenizer JSON.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_10M", "train_100M"],
        help="Splits to verify, e.g. train_10M train_100M dev test.",
    )
    parser.add_argument(
        "--show-only-mismatches",
        action="store_true",
        help="Only print rows with sentence/token mismatches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.filtered_root.exists():
        raise FileNotFoundError(f"Filtered root not found: {args.filtered_root}")
    if not args.tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {args.tokenizer_path}")

    tokenizer = get_tokenizer(args.tokenizer_path)
    results = verify(args.filtered_root, tokenizer, args.splits)

    if not results:
        print("No dataset pairs found for the requested splits.")
        raise SystemExit(2)

    print(f"filtered_root: {args.filtered_root}")
    print(f"tokenizer: {args.tokenizer_path}")
    print(f"splits: {', '.join(args.splits)}")
    print()
    print(
        f"{'dataset':<28} {'split':<12} {'abl_sent':>10} {'ctl_sent':>10} {'sent_ok':>8} "
        f"{'abl_tok':>12} {'ctl_tok':>12} {'tok_ok':>7}"
    )
    print("-" * 110)

    mismatches = 0
    for row in results:
        if args.show_only_mismatches and row.ok:
            continue

        if not row.ok:
            mismatches += 1

        print(
            f"{row.dataset:<28} {row.split:<12} "
            f"{row.ablated_sentences:>10} {row.control_sentences:>10} {str(row.sentence_match):>8} "
            f"{row.ablated_tokens:>12} {row.control_tokens:>12} {str(row.token_match):>7}"
        )

    if args.show_only_mismatches:
        mismatches = sum(1 for r in results if not r.ok)

    print()
    print(f"pairs_checked: {len(results)}")
    print(f"mismatches: {mismatches}")

    if mismatches > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
