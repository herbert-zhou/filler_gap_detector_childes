from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2TokenizerFast
import random
import argparse

os.environ["TQDM_DISABLE"] = "0"
import tqdm

SEQ_LENGTH = 64

def get_token_length(sentence: str, tokenizer):
    return len(tokenizer.encode(sentence))

def get_tokenizer():
    tokenizer_path = "../models/gpt-clean-16000.json"
    tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path), truncation=True, max_length=SEQ_LENGTH)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    return tokenizer

def display_histogram(count_dict, title, color):
    categories = list(count_dict.keys())
    counts = list(count_dict.values())

    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts, color=color)
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title(title)
    plt.xticks(rotation=45, ha='right') # Rotate labels if they overlap
    plt.tight_layout() # Adjust layout to prevent labels from being cut off
    # plt.show()
    plt.savefig(Path("./figures", title))

def display_overlay_histogram(dict_1, dict_2, title):
    x_1 = list(dict_1.keys())
    values_1 = list(dict_1.values())
    x_2 = list(dict_2.keys())
    values_2 = list(dict_2.values())
    fig, ax = plt.subplots()
    ax.bar(x_1, values_1, width=0.6, label='Total Counts', color='skyblue', alpha=0.7)
    ax.bar(x_2, values_2, width=0.6, label='Filtered Counts', color='red', alpha=0.7)
    plt.savefig(Path("./figures", title))

def filter(sentence_filter: str, split: str, filtered_tags: list, tokenizer: GPT2TokenizerFast, output_dir: str, data_dir: str, include_all: bool):
    """
        sentence_filter: the filter to use (the key in the filter dict)
        split: which split to filter on
        tokenizer: the tokenizer to use (for making token counts)
        output: where to put the filtered data to
        data_dir: the directory of the input data
        include_all: if we should include all data, or only the parent data
    """
    if include_all:
        include_dir = "all"
    else:
        include_dir = "par"

    if "dev" in split:
        output_dir_full = Path(output_dir, include_dir, sentence_filter, "dev")
        control_output_dir_full = Path(output_dir, include_dir, sentence_filter + "_control", "dev")
    elif "test" in split:
        output_dir_full = Path(output_dir, include_dir, sentence_filter, "test")
        control_output_dir_full = Path(output_dir, include_dir, sentence_filter + "_control", "test")
    elif "train_10M" in split:
        output_dir_full = Path(output_dir, include_dir, sentence_filter, "train_10M")
        control_output_dir_full = Path(output_dir, include_dir, sentence_filter + "_control", "train_10M")
    elif "train_100M" in split:
        output_dir_full = Path(output_dir, include_dir, sentence_filter, "train_100M")
        control_output_dir_full = Path(output_dir, include_dir, sentence_filter + "_control", "train_100M")

    os.makedirs(output_dir_full, exist_ok=True)
    if sentence_filter != "unfiltered":
        os.makedirs(control_output_dir_full, exist_ok=True)

    base = f"LABELED_{split}.csv"
    fw = open(Path(output_dir_full, f"{split}.txt"), "w")

    total_count_dict = {}
    filtered_count_dict = {}
    sentence_dict = {}   
    non_filtered_sentence_dict = {}

    # There are no sentences with length over 256 
    MAX_LEN = 256        
    for i in range(MAX_LEN):
        total_count_dict[i] = 0
        filtered_count_dict[i] = 0
        sentence_dict[i] = []
        non_filtered_sentence_dict[i] = []

    filtered_sentences = []
    non_filtered_sentences = []
    
    data_file_path = Path(data_dir, base)
    if not os.path.exists(data_file_path):
        return
    df = pd.read_csv(data_file_path)
    for index, row in tqdm.tqdm(df.iterrows()):
        if include_all == False and row["speaker"] == "CHI":
            # we want to consider only utterances spoken by parents
            continue
        keep = True
        token_length = get_token_length(row["sentence_clean"], tokenizer)
        total_count_dict[token_length] += 1
        sentence_dict[token_length] += [row["sentence_clean"]]
        
        # Check if sentence has the current filter's tags
        for tag in filtered_tags:
            if tag in row["labels"]:
                # if we see any part of the filter, we do not keep 
                keep = False
                break

        if (keep == False):
            filtered_sentences += [row["sentence_clean"]]
            filtered_count_dict[token_length] += 1
        else:
            non_filtered_sentences += [row["sentence_clean"]]
            non_filtered_sentence_dict[token_length] += [row["sentence_clean"]]

    random.shuffle(non_filtered_sentences)
    for sentence in non_filtered_sentences:
        fw.write(sentence + "\n")

    if sentence_filter == "unfiltered":
        return

    # Control: start from the FULL pool of sentences (same as the ablated 
    # experiment starts from), and randomly remove the same number of 
    # sentences per token-length bin as the filter removed. This ensures
    # ablated and control datasets are the same size, differing only in
    # WHICH sentences were removed (targeted vs. random).
    control_sentences = []            
    for length in sentence_dict.keys():
        all_sentences_at_length = sentence_dict[length]
        count_at_length = len(all_sentences_at_length)
        # Remove the same number of sentences as the filter removed at this length
        remove_count = min(filtered_count_dict[length], count_at_length)
        remove_index_set = set(random.sample(range(count_at_length), remove_count))

        for i in range(count_at_length):
            if i not in remove_index_set:
                control_sentences += [all_sentences_at_length[i]]
    random.shuffle(control_sentences)

    fw_control = open(Path(control_output_dir_full, f"{split}.txt"), "w")
    for sentence in control_sentences:
        fw_control.write(sentence + "\n")

    # display_histogram(total_count_dict, f"total_counts_{split}", "skyblue")
    # display_histogram(filtered_count_dict, f"filtered_counts_{sentence_filter}_{split}", "red")
    display_overlay_histogram(total_count_dict, filtered_count_dict, f"filtered_counts_{sentence_filter}_{split}")


filter_dict = {
    "unfiltered": [],
    "matrix_questions": ["SMQ", "OMQ", "CC_SMQ", "CC_OMQ", "AMQ", "PMQ", "PlainMQ"],
    "embedded_questions": ["SEQ", "OEQ", "AEQ", "PEQ"],
    "relative_clauses": ["SRC", "ORC", "SRC_reduced", "ORC_reduced", "ARC", "PRC"],
    # "subject_matrix_question": ["SMQ"],
    # "object_matrix_question": ["OMQ"],
    # "adjunct_matrix_question": ["AMQ"],
    # "polar_matrix_question": ["PMQ"],
    # "subject_embedded_question": ["SEQ"], #clean
    # "object_embedded_question": ["OEQ"], #clean
    # "adjunct_embedded_question": ["AEQ"], #clean? -> do not generalize?
    # "polar_embedded_question": ["PEQ"], #clean? -> do not generalize?
    # "subject_relative_clause": ["SRC"],
    # "object_relative_clause": ["ORC"],
    # "adjunct_relative_clause": ["ARC"],
    # "reduced_subject_relative_clause": ["SRC_reduced"],
    # "reduced_object_relative_clause": ["ORC_reduced"],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_filter", type=str, default=None, help="Filter key from filter_dict")
    parser.add_argument("--split", type=str, default=None, help="Split to process")
    parser.add_argument("--include", type=str, default="par", choices=["par", "all"], help="Include parents only (par) or all")
    parser.add_argument("--data_dir", type=str, default="./data/labeled/", help="Directory with labeled CSVs")
    parser.add_argument("--output_dir", type=str, default="./data/filtered/", help="Directory to write filtered data")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    include_all = args.include == "all"

    splits = ["dev", "test", "train_10M", "train_100M_part1", "train_100M_part2", "train_100M_part3", "train_100M_part4", "train_100M_part5", "train_100M_part6"]
    filters = list(filter_dict.keys())
    tokenizer = get_tokenizer()

    if args.sentence_filter or args.split:
        if not args.sentence_filter or not args.split:
            raise ValueError("Both --sentence_filter and --split must be provided together.")
        if args.sentence_filter not in filter_dict:
            raise ValueError(f"Unknown sentence_filter: {args.sentence_filter}")
        if args.split not in splits:
            raise ValueError(f"Unknown split: {args.split}")
        filtered_tags = filter_dict[args.sentence_filter]
        filter(
            sentence_filter=args.sentence_filter,
            split=args.split,
            filtered_tags=filtered_tags,
            tokenizer=tokenizer,
            data_dir=data_dir,
            output_dir=output_dir,
            include_all=include_all,
        )
    else:
        for sentence_filter in filters:
            print(f"sentence_filter: {sentence_filter}")
            filtered_tags = filter_dict[sentence_filter]
            for split in splits:
                filter(
                    sentence_filter=sentence_filter,
                    split=split,
                    filtered_tags=filtered_tags,
                    tokenizer=tokenizer,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    include_all=include_all,
                )