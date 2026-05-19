# Reproduction of BabyLM Experiment (Section 6 of the paper)

### 1. Label BabyLM utterances

`detection_algorithms/parse_babylm_labels.py` is a sharded batch driver that applies the unified detector wrapper to BabyLM CHILDES utterances:

```bash
python detection_algorithms/parse_babylm_labels.py --dataset train_10M --start-idx 0
python detection_algorithms/parse_babylm_labels.py --dataset train_10M --start-idx 0 --child-filtered
```

Inputs are read from `datasets/BabuLM/text_data/<split>/childes.<suffix>`. Outputs are CSV shards in `BabyLM_experiment/BabyLM_dataset_labeled/` with columns `sentence`, `sentence_clean`, `speaker`, and `labels`. After all shards finish, merge them into the `BabyLM_experiment/data/labeled/LABELED_<split>.csv` schema expected by the downstream code.

### 2. Build filtered and control corpora

`BabyLM_experiment/src/data_processor.py` reads `BabyLM_experiment/data/labeled/LABELED_<split>.csv` and produces, for each of three filters (`matrix_questions`, `embedded_questions`, `relative_clauses`):

- a **filtered** corpus with every sentence carrying any tag in that filter removed, and
- a **control** corpus that randomly removes the same number of sentences per token-length bin so the two corpora match in sentence count and length distribution.

Run a single combination:

```bash
cd BabyLM_experiment/src
python data_processor.py \
    --sentence_filter matrix_questions \
    --split train_10M \
    --include par \
    --data_dir   ../data/labeled/ \
    --output_dir ../data/filtered/
```

Or submit the full sweep over `{unfiltered, matrix_questions, embedded_questions, relative_clauses} × {splits}` as a SLURM array (36 jobs):

```bash
cd BabyLM_experiment/scripts
sbatch data_processor.sh
```

Outputs land in `BabyLM_experiment/data/filtered/par/<filter>/<split>/<split>.txt` and `BabyLM_experiment/data/filtered/par/<filter>_control/<split>/<split>.txt`. For the 100M shards, `data_processor.py` writes the part files under the shared `train_100M/` directory, e.g. `BabyLM_experiment/data/filtered/par/<filter>/train_100M/train_100M_part1.txt`.

After this step, sanity-check that ablated and control corpora are the same size:

```bash
cd BabyLM_experiment/src
python verify_control_tokens.py
python count_training_tokens.py
```

### 3. Train language models

`BabyLM_experiment/src/train.py` reads a YAML config and one filter/control pair, tokenizes with `BabylmDataset`, and trains a GPT-2 or LLaMA causal LM. Single run:

```bash
cd BabyLM_experiment/src
python train.py \
    --config ../config/llama-360M.yaml \
    --filter matrix_questions \
    --size 10M \
    --include par \
    --seed 0
# add --control for the matched-control run
```

To reproduce the paper's 15-seed × 3-filter × {filtered, control} + unfiltered sweep on the cluster:

```bash
cd BabyLM_experiment/scripts
sbatch train_llama_10M.sh   # LLaMA 360M, array 0-104
sbatch train_gpt_10M.sh     # GPT-2 705M
```

Hyperparameters are taken from `BabyLM_experiment/config/llama-360M.yaml` and `BabyLM_experiment/config/gpt-705M.yaml` and match Table 7 of the paper (seq length 128, batch size 128, ~10 epochs, fp16, lr `3e-4` / `2.5e-4`). Trained checkpoints land in `BabyLM_experiment/models/par/<filter>/<model>-<size>[-control]/lr<lr>_wd<wd>/seed_<seed>/final/`.

### 4. Evaluate on minimal pairs

Minimal pairs are generated and scored inside `BabyLM_experiment/src/analysis.py` from the 15 templates described in Appendix H. For one (filter, model, seed, category) cell:

```bash
cd BabyLM_experiment/src
python analysis.py \
    --dataset_filter matrix_questions \
    --model_config llama-360M \
    --dataset_size 10M \
    --include_dir par \
    --seed 0 \
    --minimal_pair_category gap_animate_matrix
# add --control to evaluate the matched-control model
```

The full sweep (3 filters × 2 model families × 15 seeds × 32 categories × {filtered, control}) is submitted via:

```bash
cd BabyLM_experiment/scripts
sbatch analysis.sh
```

Per-cell accuracies are written under `BabyLM_experiment/analysis_data/`. Validate evaluation-set sizes with:

```bash
python count_minimal_pair_sentences.py
```

### 5. Aggregate and plot

```bash
cd BabyLM_experiment/src
python aggregate_results.py        # merges per-seed/per-category CSVs
python generate_figures.py         # figures 9 + appendix bar plots
python plot_loss_curves.py         # eval-loss curves from trainer logs
```

Figures are written to `BabyLM_experiment/figures/` and aggregate tables to `BabyLM_experiment/analysis_data/`.

---

## Expected results

With the configuration above (15 seeds, BabyLM CHILDES 10M, parent-only utterances), filtered-vs-control minimal-pair accuracies should reproduce the qualitative pattern in Figure 9 / Figure 12 of the paper:

- Filtering each construction significantly hurts accuracy on that same construction.
- Filtering **matrix questions** additionally degrades embedded-question and relative-clause accuracy, while filtering the other two constructions does not transfer back to matrix questions.

Exact numbers will vary slightly with library versions and GPU determinism.
