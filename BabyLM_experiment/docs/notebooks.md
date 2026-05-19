# Notebook Usage

The notebooks in `notebooks/` are optional reproducibility and audit aids. They are not required for the main filtered-corpus training pipeline in `README.md`, but they document two preprocessing analyses that were useful around the experiments:

- `notebooks/cleaning_and_tokenization.ipynb`: clean raw BabyLM text files and train the shared byte-level BPE tokenizer.
- `notebooks/check_lexical_frequency.ipynb`: count how often evaluation-set lexical items appear in labeled CHILDES/BabyLM data.

These notebooks were copied from an older working tree, so check their path cells before running them. The examples below assume the notebook kernel's working directory is `filler_gap_detector/src`. If your kernel starts elsewhere, either change to `src` first or adjust the relative paths accordingly.

```python
from pathlib import Path
Path.cwd()
```

## Environment

Use the same `ling-gen` environment described in the main README. The cleaning/tokenizer notebook also uses the `tokenizers` package, which is normally installed as a dependency of `transformers`. The lexical-frequency notebook imports spaCy's small English model, so install it if needed:

```bash
python -m spacy download en_core_web_sm
```

## `cleaning_and_tokenization.ipynb`

Use this notebook only if you need to rebuild cleaned text files or regenerate `models/gpt-clean-16000.json`.

Before running the cleaning cells, update the old absolute `DATA_ROOT` assignment to the local BabyLM text directory:

```python
DATA_ROOT = Path("../BabyLM/text_data").resolve()
```

The cleaning section expects split directories such as:

```text
BabyLM/text_data/train_10M/childes.train
BabyLM/text_data/dev/childes.dev
```

It writes cleaned versions next to the raw splits, for example:

```text
BabyLM/text_data/train_10M_clean/childes.train
BabyLM/text_data/dev_clean/childes.dev
```

The tokenizer-training section fits a byte-level BPE tokenizer on the cleaned training files with these settings:

- `vocab_size=16000`
- `min_frequency=2`
- special tokens: `<pad>`, `<s>`, `</s>`
- ByteLevel pre-tokenizer, decoder, and post-processor
- `NFKC` normalization

To make the generated tokenizer usable by the rest of this repository, save it at the path expected by `src/train.py`, `src/data_processor.py`, and the QA scripts:

```python
tokenizer_path = Path("../models/gpt-clean-16000.json").resolve()
tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(tokenizer_path), pretty=True)
```

After saving, run the final test cell to confirm that encoding and decoding work. Do not commit regenerated cleaned text or tokenizer artifacts unless you explicitly intend to version those data products.

## `check_lexical_frequency.ipynb`

Use this notebook to audit whether lexical items used in the synthetic minimal pairs are frequent enough in the labeled data.

Before running it in the cleaned repository, update the old labeled-data path inside `load_dataset`. If running from `src`, use:

```python
file_path = f"../data/labeled/LABELED_{dataset}.csv"
```

The main counting cells produce JSON files with counts for evaluation verbs by construction condition. To keep outputs at the repository root rather than inside `src`, use output paths such as:

```python
json_path = f"../dataset_lexical_frequency/EvalTokens_{dataset}_{file_child}.json"
```

The notebook distinguishes child vs. parent speech with this naming convention:

- `Child`: rows where `speaker == "CHI"`
- `noChild`: rows where `speaker == "PAR"`

The top-level construction groups are:

- `MQ`: `SMQ`, `OMQ`, `CC_SMQ`, `CC_OMQ`
- `EQ`: `SEQ`, `OEQ`
- `RC`: `SRC`, `ORC`, `SRC_reduced`, `ORC_reduced`

The archived cells at the bottom contain older tokenizer-based and lemma-based counting experiments. Treat those as exploratory checks rather than part of the main pipeline.

## Relationship To Source Files

Some notebook logic already exists in source form:

- `src/mrclean.py` contains the cleanup functions imported by `cleaning_and_tokenization.ipynb`.
- `src/check_lexical_frequency.py` mirrors the lexical-frequency notebook logic, but still follows the older hardcoded path conventions.

For reproducible full-pipeline runs, prefer the scripts documented in the README. Use the notebooks when you need to inspect, regenerate, or explain preprocessing artifacts interactively.