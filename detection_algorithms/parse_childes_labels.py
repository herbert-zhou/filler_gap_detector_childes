import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

########################################################################
# Get dataset
########################################################################
SENTENCE_TYPES = {
    "declarative": ".",
    "question": "?",
    "trail off": " ...",
    "interruption": "",
    "trail off question": "?",
    "imperative_emphatic": "!",
    "interruption question": "?",
    "quotation next line": ":",
    "self interruption": "...",
    "quotation precedes": ".",
    "self interruption question": " xxx?",
    "broken for coding": ".",
    "missing CA terminator": "",
    "question exclamation": "!",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run filler-gap detectors over a batch of processed CHILDES utterances."
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        required=True,
        help="Zero-based batch index to label.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=180000,
        help="Number of rows to process per batch. Default: 180000.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=ROOT_DIR / "childes_statistics" / "Processed_Utterances_Updated.csv",
        help="Input processed utterance CSV. Defaults to ROOT_DIR/childes_statistics/Processed_Utterances_Updated.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "childes_statistics" / "childes_statistics_LABELED",
        help="Directory for labeled batch CSVs. Defaults to ROOT_DIR/childes_statistics/childes_statistics_LABELED.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd
    from tqdm.auto import tqdm

    from detection_algorithms.get_labels import get_labels

    input_file = args.input_file if args.input_file.is_absolute() else ROOT_DIR / args.input_file
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT_DIR / args.output_dir

    df = pd.read_csv(input_file)
    start = args.start_idx * args.batch_size
    stop = min((args.start_idx + 1) * args.batch_size, df.shape[0])
    df_sub = df.iloc[start:stop]
    df_sub = df_sub[~df_sub["gloss"].isna()].reset_index(drop=True)

    df_sub["sentence_mod"] = [
        row["gloss"] + SENTENCE_TYPES.get(row["type"], "")
        for _, row in df_sub.iterrows()
    ]

    utterances = df_sub["sentence_mod"].tolist()
    print(f"Input file: {input_file}")
    print(f"Batch index: {args.start_idx} | Rows: {start}:{stop}")
    print(f"Total number of non-empty sentences: {len(utterances)}")

    parser_labels = []
    for _, sen in enumerate(tqdm(utterances, miniters=1000), 1):
        labels = get_labels(sen)
        parser_labels.append(labels)
    df_sub["parser_labels"] = parser_labels

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"All_LABELED_{args.start_idx}.csv"
    df_sub.to_csv(output_file, index=False)
    print(f"Wrote labeled batch to: {output_file}")


if __name__ == "__main__":
    main()
