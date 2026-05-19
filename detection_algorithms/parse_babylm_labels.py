import argparse
import csv
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run filler-gap detectors over a BabyLM CHILDES split."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="BabyLM split name, e.g. dev, test, train_100M, or train_10M.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        required=True,
        help="Zero-based batch index to label.",
    )
    parser.add_argument(
        "--child-filtered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When set, label non-CHI utterances only. Default labels CHI utterances only.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT_DIR / "datasets" / "BabuLM",
        help="BabyLM data root. Defaults to ROOT_DIR/datasets/BabuLM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "BabyLM_experiment" / "BabyLM_dataset_labeled",
        help="Output directory for labeled CSV batches. Defaults to ROOT_DIR/BabyLM_experiment/BabyLM_dataset_labeled.",
    )
    return parser.parse_args()


def remove_brackets(sentence: str) -> str:
    cleaned = re.sub(r"\[.*?\]", "", sentence)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
    return cleaned


def load_utterances(file_path: Path, child_filtered: bool) -> list[str]:
    with file_path.open("r", encoding="utf-8") as f:
        raw = f.readlines()

    conversations = [r for r in raw if r.startswith("*")]
    if child_filtered:
        return ["\t".join(c.split("\t")[1:]).strip() for c in conversations if not c.startswith("*CHI:")]
    return ["\t".join(c.split("\t")[1:]).strip() for c in conversations if c.startswith("*CHI:")]


def batch_size(dataset: str, n_utterances: int) -> int:
    if dataset == "train_100M":
        return 186530
    return n_utterances if n_utterances < 200000 else int(n_utterances / 2)


def main() -> None:
    args = parse_args()

    from tqdm.auto import tqdm

    from detection_algorithms.get_labels import get_labels

    data_root = args.data_root if args.data_root.is_absolute() else ROOT_DIR / args.data_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT_DIR / args.output_dir

    suffix = args.dataset.split("_")[0]
    file_path = data_root / "text_data" / args.dataset / f"childes.{suffix}"
    file_children = "noChild" if args.child_filtered else "Child"
    output_path = output_dir / f"{args.dataset}_{file_children}_{args.start_idx}.csv"

    print(f"File path = {file_path}")
    print(f"Output path = {output_path}")

    utterances = load_utterances(file_path, args.child_filtered)
    print(f"Total number of sentences: {len(utterances)}")

    bs = batch_size(args.dataset, len(utterances))
    start = args.start_idx * bs
    stop = min((args.start_idx + 1) * bs, len(utterances))
    batch = utterances[start:stop]
    print(f"Batch index: {args.start_idx} | Rows: {start}:{stop}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sentence", "sentence_clean", "speaker", "labels"],
        )
        writer.writeheader()

        for sen in tqdm(batch, miniters=1000):
            sen_clean = remove_brackets(sen)
            labels = get_labels(sen_clean)
            writer.writerow(
                {
                    "sentence": sen,
                    "sentence_clean": sen_clean,
                    "speaker": "PAR" if args.child_filtered else "CHI",
                    "labels": labels,
                }
            )

    print(f"Wrote labeled batch to: {output_path}")


if __name__ == "__main__":
    main()
