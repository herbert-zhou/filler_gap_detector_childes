from pathlib import Path
import argparse
import csv
from typing import Any

from transformers import GPT2TokenizerFast


FILTER_TAGS = {
    "matrix_questions": ["SMQ", "OMQ", "CC_SMQ", "CC_OMQ", "AMQ", "PMQ", "PlainMQ"],
    "embedded_questions": ["SEQ", "OEQ", "AEQ", "PEQ"],
    "relative_clauses": ["SRC", "ORC", "SRC_reduced", "ORC_reduced", "ARC", "PRC"],
}


FILTER_MAP = {
    "matrix": "matrix_questions",
    "embedded": "embedded_questions",
    "relative": "relative_clauses",
    "matrix_questions": "matrix_questions",
    "embedded_questions": "embedded_questions",
    "relative_clauses": "relative_clauses",
    "unablated": "unablated",
    "unfiltered": "unablated",
}


def get_tokenizer(tokenizer_path: Path) -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    return tokenizer


def get_split_csv_paths(labeled_data_path: Path, size: str) -> list[Path]:
    if size == "10M":
        return [labeled_data_path / "LABELED_train_10M.csv"]
    if size == "100M":
        return [
            labeled_data_path / f"LABELED_train_100M_part{part}.csv"
            for part in range(1, 7)
        ]
    raise ValueError("--size must be one of: 10M, 100M")


def should_keep_sentence(labels: str, filter_name: str) -> bool:
    if filter_name == "unablated":
        return True

    for tag in FILTER_TAGS[filter_name]:
        if tag in labels:
            return False
    return True


def count_from_csvs(
    csv_paths: list[Path],
    tokenizer: GPT2TokenizerFast,
    filter_name: str,
    include: str,
) -> tuple[int, int]:
    sentence_count = 0
    token_count = 0

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {csv_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if include == "par" and row.get("speaker") == "CHI":
                    continue

                sentence = row.get("sentence_clean", "")
                labels = row.get("labels", "")

                if not sentence:
                    continue
                if not should_keep_sentence(labels=labels, filter_name=filter_name):
                    continue

                # Special tokens are explicitly excluded.
                encoded = tokenizer(sentence, add_special_tokens=False)
                token_count += len(encoded["input_ids"])
                sentence_count += 1

    return sentence_count, token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count sentence and token totals for ablated datasets + unablated dataset "
            "from labeled CSVs, excluding special tokens."
        )
    )
    parser.add_argument(
        "--labeled-data-path",
        type=Path,
        default=Path("./data/labeled"),
        help="Path to labeled CSV files.",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="par",
        help="Subset under data/filtered to use (e.g., par or all).",
        choices=["par", "all"],
    )
    parser.add_argument(
        "--size",
        type=str,
        default="10M",
        choices=["10M", "100M"],
        help="Training size to count.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("./models/gpt-clean-16000.json"),
        help="Path to tokenizer json.",
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["matrix", "relative", "embedded", "unablated"],
        help="Filters to count. Accepts short names and includes unablated/unfiltered.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = get_tokenizer(args.tokenizer_path)
    csv_paths = get_split_csv_paths(args.labeled_data_path, args.size)

    canonical_filters = []
    for filter_name in args.filters:
        if filter_name not in FILTER_MAP:
            valid = ", ".join(sorted(FILTER_MAP.keys()))
            raise ValueError(f"Unknown filter '{filter_name}'. Valid options: {valid}")
        canonical_filters.append(FILTER_MAP[filter_name])

    unablated_sentence_count, unablated_token_count = count_from_csvs(
        csv_paths=csv_paths,
        tokenizer=tokenizer,
        filter_name="unablated",
        include=args.include,
    )

    results: list[dict[str, Any]] = []
    for filter_name in canonical_filters:
        sentence_count, token_count = count_from_csvs(
            csv_paths=csv_paths,
            tokenizer=tokenizer,
            filter_name=filter_name,
            include=args.include,
        )
        sentences_ablated = max(unablated_sentence_count - sentence_count, 0)
        tokens_ablated = max(unablated_token_count - token_count, 0)
        results.append(
            {
                "dataset": filter_name,
                "sentences": sentence_count,
                "tokens_no_special": token_count,
                "sentences_ablated": sentences_ablated,
                "tokens_ablated": tokens_ablated,
            }
        )

    print(f"Training size: {args.size}")
    print(f"Include set: {args.include}")
    print("Source CSVs:")
    for csv_path in csv_paths:
        print(f"  - {csv_path}")
    print()
    print(
        f"{'dataset':<20} {'sentences':>12} {'tokens_no_special':>20} {'sentences_ablated':>20} {'tokens_ablated':>16}"
    )
    print("-" * 92)

    for result in results:
        print(
            f"{result['dataset']:<20} {result['sentences']:>12} "
            f"{result['tokens_no_special']:>20} {result['sentences_ablated']:>20} "
            f"{result['tokens_ablated']:>16}"
        )

    output_csv_path = Path("figures/counts.csv")
    with output_csv_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "dataset",
                "sentences",
                "tokens_no_special",
                "sentences_ablated",
                "tokens_ablated",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Wrote counts to: {output_csv_path}")


if __name__ == "__main__":
    main()