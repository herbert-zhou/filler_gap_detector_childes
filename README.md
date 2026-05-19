# filler_gap_detector_childes

Code and data-processing workflows for the CoNLL 2026 paper: [*What Exactly do Children Receive in Language Acquisition? A Case Study on CHILDES with Automated Detection of Filler-Gap Dependencies*](https://arxiv.org/pdf/2603.02082).

This project provides automated tools for identifying filler-gap dependencies (FGDs) in English child-language corpora. The detectors target three core construction families: matrix wh-questions, embedded wh-questions, and relative clauses. Within each family, the code assigns finer-grained labels by extraction site or subtype, including subject, object, adjunct, polar, possessive, and reduced relative-clause categories where applicable.

The repository has three main components:

1. **Detection algorithms** for labeling FGD constructions in individual sentences, using complementary information from dependency and constituency parses.
2. **CHILDES corpus processing and statistics** for preparing transcript/utterance metadata, applying the detectors to CHILDES, comparing against the CHILDES Treebank, and reproducing descriptive corpus figures.
3. **BabyLM filtered-corpus experiments** for labeling BabyLM CHILDES utterances, constructing FGD-ablated and matched-control corpora, training language models, and evaluating syntactic generalization on minimal pairs.

The `dataset` directory contains raw datasets such as CHILDES, CHILDES Treebank, and BabyLM, all accessed online (described below). These data are governed by their original licenses and may need to be downloaded separately. 

**For datasets labeled by our detector**, see:
- `filler_gap_detector_childes/childes_statistics/childes_statistics_LABELED` for CHILDES;
- `filler_gap_detector_childes/BabyLM_experiemnt/BabyLM_dataset_labeled` for BabyLM text datasets.

---

## 1. Repository layout

```
filler_gap_detector_childes/
├── detection_algorithms/   # FGD detectors and batch labeling drivers
├── childes_statistics/     # CHILDES preprocessing, labeling outputs, statistics, and figures
├── BabyLM_experiment/      # Filtered-corpus language-model experiments
├── datasets/               # Local CHILDES / Treebank / BabyLM inputs, not all intended for release
└── README.md
```

One-level contents of the main code directories:

```
detection_algorithms/
├── MatrixQ_Detector.py     # matrix-question detector
├── EmbQ_Detector.py        # embedded-question detector
├── RC_Detector.py          # relative-clause detector
├── get_labels.py           # unified sentence-level labeling wrapper
├── parse_childes_labels.py # batch labeling for processed CHILDES utterances
└── parse_babylm_labels.py  # batch labeling for BabyLM CHILDES utterances

childes_statistics/
├── childes_processing.ipynb        # CHILDES transcript/utterance preprocessing
├── childes_stats_notes.md          # preprocessing notes and reproducibility details
├── plotting.ipynb                  # corpus statistics and figure generation
├── compare_to_treebank/            # comparison against CHILDES Treebank annotations
├── childes_statistics_LABELED/     # detector-labeled CHILDES utterance shards
└── figures/                        # generated corpus-statistics figures

BabyLM_experiment/
├── src/                    # corpus filtering, training, evaluation, and aggregation code
├── scripts/                # SLURM launch scripts
├── config/                 # GPT-2 / LLaMA training configs
├── docs/                   # BabyLM experiment reproduction notes
├── notebooks/              # optional preprocessing and audit notebooks
└── BabyLM_dataset_labeled/ # detector-labeled BabyLM CSV shards
```

The `datasets/` directory is used as the local location for externally obtained corpora, including CHILDES, the CHILDES Treebank, and BabyLM files. These data are governed by their original licenses and may need to be downloaded separately.

---

## 2. Environment

The pipeline depends on spaCy (transformer model), benepar, NLTK, PyTorch, HuggingFace `transformers`, `datasets`, `pandas`, `matplotlib`, `tqdm`, and `wandb` (optional). The optional tokenizer notebook also uses HuggingFace `tokenizers`.

A minimal conda environment:

```bash
conda create -n ling-gen python=3.10 -y
conda activate ling-gen

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install \
    "transformers>=4.40" datasets accelerate \
    spacy benepar nltk \
    pandas numpy matplotlib seaborn tqdm pyyaml wandb

python -m spacy download en_core_web_trf
python -c "import benepar; benepar.download('benepar_en3_large')"
python -c "import nltk; nltk.download('punkt')"
```

The SLURM scripts all `conda activate ling-gen`. A GPU is required for training and for benepar at scale; CPU is fine for small-scale detector sanity checks.

---

## 3. Input data

This repository expects externally distributed corpora to be downloaded separately and placed under `datasets/`.

For the CHILDES corpus statistics and large-scale detector labeling, download all English North American CHILDES corpora from [CHILDES Eng-NA](https://talkbank.org/childes/access/Eng-NA/) and the corresponding English North American PhonBank corpora from [PhonBank Eng-NA](https://talkbank.org/phon/access/Eng-NA/). Save the combined local CHAT files under:

```
filler_gap_detector_childes/
└── datasets/
    └── CHILDES-Eng_NA/
        ├── <CHILDES corpus directories>
        └── <PhonBank corpus directories>
```

For the comparison with human-annotated Treebank data, download the CHILDES Treebank from [Pearl & Sprouse's CHILDES Treebank page](https://sites.socsci.uci.edu/~lpearl/CoLaLab/CHILDESTreebank/childestreebank.html) and place it under:

```
filler_gap_detector_childes/
└── datasets/
    └── CHILDESTreebank-curr/
```

For the BabyLM filtered-corpus experiments, download the BabyLM Challenge CHILDES splits from [https://babylm.github.io/](https://babylm.github.io/) and lay out the raw text files as:

```
filler_gap_detector_childes/
└── datasets/
    └── BabuLM/
        └── text_data/
            ├── dev/childes.dev
            ├── test/childes.test
            ├── train_10M/childes.train
            └── train_100M/childes.train     # only for 100M experiments
```

The BabyLM labeling step expects one CHAT-style utterance per line starting with `*SPEAKER:`. Labeled CSV shards are written to `BabyLM_experiment/BabyLM_dataset_labeled/` with columns `sentence`, `sentence_clean`, `speaker`, and `labels`. After all shards finish, merge them into the `BabyLM_experiment/data/labeled/LABELED_<split>.csv` schema expected by the downstream filtered-corpus code.

---



## 4. Running the detector

The main detector code is in `detection_algorithms/`. The three construction-specific detector files are:

- `MatrixQ_Detector.py`: matrix wh-question and polar-question detection.
- `EmbQ_Detector.py`: embedded question detection.
- `RC_Detector.py`: relative clause detection.

For most use cases, run the unified wrapper in `get_labels.py`. Given any sentence, `get_labels(sentence)` returns a list of relevant FGD labels, if any:

```python
from detection_algorithms.get_labels import get_labels

labels = get_labels("Who did the professor praise?")
print(labels)
```

Batch labeling scripts call the same wrapper:

- `parse_childes_labels.py` labels processed CHILDES utterance batches.
- `parse_babylm_labels.py` labels BabyLM CHILDES utterance batches.

---

## 5. Citations

If you use this code or the resulting annotations, please cite:

```bibtex
@inproceedings{zhou2026fillergap,
  title     = {What Exactly do Children Receive in Language Acquisition?
               A Case Study on CHILDES with Automated Detection of
               Filler-Gap Dependencies},
  author    = {Zhou, Zhenghao Herbert and Dai, William and Viswanathan, Maya
               and Charlow, Simon and McCoy, R. Thomas and Frank, Robert},
  booktitle = {Proceedings of the 30th Conference on Computational Natural
               Language Learning (CoNLL)},
  year      = {2026}
}
```

---

## 6. License

Code is released under the terms of the [LICENSE](LICENSE) file in this repository. CHILDES, CHILDES Treebank, and BabyLM data are subject to their own licenses; obtain them from their original sources.
