# CHILDES Statistics Preprocessing Notes

This document summarizes the preprocessing workflow used to prepare the CHILDES Eng-NA utterance data for the statistical analyses in the paper. The goal of the preprocessing step is to produce transcript-level metadata and utterance-level tables with consistent speaker-role and age information, so that downstream analyses can aggregate filler-gap dependency counts by child age, speaker type, corpus type, and activity type. The relevant code is in `childes_statistics/childes_processing.ipynb`.

## Input Data

The preprocessing uses two complementary sources of CHILDES information:

1. `All_Transcripts.csv` and `All_Utterances.csv`, obtained from the `childesr` API.
2. A local download of the corresponding CHAT files under `CHILDES-Eng-NA/`, including `.cha` transcript files and `0types.txt` metadata files.

The `childesr` tables provide convenient transcript- and utterance-level data. The local CHAT files are needed to recover metadata that is not fully available or not consistently represented in the API output, especially corpus activity types and speaker-specific age information.

## 1. Create Transcript-Level Metadata

The first step loads `All_Transcripts.csv`, keeps only rows from the Eng-NA collection, and checks that the resulting table contains 57 unique corpora. Columns not needed for downstream analysis are removed, including language, collection identifiers, date, target child sex, and other API metadata.

The resulting reduced transcript table is saved as:

```text
Processed_Transcripts.csv
```

This file preserves the fields needed to link utterances to local CHAT files, including `transcript_id`, `corpus_name`, `filename`, `target_child_name`, and `target_child_age`.

## 2. Annotate Transcripts with Study Type and Activity

Each transcript is then annotated with two additional metadata fields:

- `study_type`: the corpus design, such as longitudinal or cross-sectional.
- `activity`: the communicative setting or task, such as toy play, meal, book reading, writing, or testing.

For each transcript, the API filename is converted from the `childesr` XML-style path to the corresponding local CHAT path. For example:

```text
Eng-NA/Garvey/amyann.xml
```

is mapped to:

```text
CHILDES-Eng-NA/Garvey/amyann.cha
```

Starting from the `.cha` file's directory, the script searches upward within the corpus directory for the nearest `0types.txt` file. The first non-empty `@Types:` line is parsed, and the first two comma-separated fields are used as `study_type` and `activity`.

The notebook output reported that 104 unique `0types.txt` files were used. The resulting activity distribution included:

| Activity | Transcripts |
| --- | ---: |
| toyplay | 6,096 |
| preverbal | 1,610 |
| narrative | 839 |
| adult | 432 |
| meal | 314 |
| tests | 311 |
| book | 292 |
| everyday | 197 |
| writing | 125 |
| group | 122 |
| reading | 32 |
| pictures | 26 |

Several missing or irregular type files were checked manually:

- `Haggerty` has no `0types.txt`; because it contains only one `.cha` file, it was treated as cross-sectional and everyday/conversation-like.
- `HSLLD/HV5/LW` and `HSLLD/HV7/LW` do not have local type files. `LW` was interpreted as letter writing, and these files were treated as longitudinal writing data.
- `HSLLD/HV7/MD` does not have a local type file. Based on the TalkBank documentation, `MD` was interpreted as mother definition, an experimental task, and was treated as longitudinal everyday/experimental data.

## 3. Inspect Speaker Roles and Age Metadata

The next step scans all local `.cha` files under `CHILDES-Eng-NA/` and parses their `@ID:` lines. CHAT `@ID:` lines follow this general schema:

```text
@ID: language|corpus|code|age|sex|group|eth,SES|role|education|custom|
```

The script extracts speaker code, age, and role from each `@ID:` line. CHAT age strings are converted to months. For example:

- `2;11.07` is converted to 35 months.
- `2;04.` is converted to 28 months.
- `34;` is converted to 408 months.

This scan is used to determine which speaker roles should be treated as child speech and which should be treated as adult/input speech.

The scan produced the following summary:

```text
Total @ID lines:                    26,838
Target_Child @ID lines:             10,096
Non-Target_Child @ID with AGE:       1,035

Target_Child-role @ID lines:
  Total role == Target_Child:        10,096
  Target_Child with age:              9,349

Child-role @ID lines:
  Total role == Child:                  747
  Child with age filled:                170
```

For non-target-child roles with age filled, the main role counts and mean ages in months were:

| Role | Count | Mean Age Months |
| --- | ---: | ---: |
| Mother | 359 | 312.06 |
| Investigator | 173 | 312.72 |
| Child | 170 | 43.02 |
| Father | 129 | 313.40 |
| Media | 58 | 310.34 |
| Adult | 35 | 311.66 |
| Grandmother | 31 | 609.68 |
| Brother | 27 | 140.89 |
| Environment | 25 | 307.20 |
| Sister | 8 | 126.38 |
| Unidentified | 8 | 312.00 |
| Friend | 5 | 261.60 |
| Visitor | 4 | 312.00 |
| Grandfather | 2 | 612.00 |
| Relative | 1 | 300.00 |

Based on this distribution, the preprocessing treats `Target_Child` and `Child` as child speech. All other roles are treated as adult/input speech. Although some roles such as `Brother` and `Sister` may refer to children in ordinary usage, the filled ages for these roles in the data are generally much older than the target-child age range relevant to this analysis.

## 4. Export Known Non-Target Child Ages & Link Non-Target Child Ages to Transcript IDs

The 170 `Child` role speakers with filled ages are exported to:

```text
child_role_with_age.csv
```

This table records:

- `filename`: local CHAT filename.
- `corpus_name`: corpus name from the `@ID:` line.
- `child_code`: speaker code for the non-target child.
- `age`: speaker age in months.

These rows are needed because utterances from `Child` speakers are child speech, but their age is not necessarily the same as the target child's age for the transcript.

The local CHAT filenames in `child_role_with_age.csv` are normalized to match the filename convention in `Processed_Transcripts.csv`. For example:

```text
CHILDES-Eng-NA/Providence/Alex/011006.cha
```

is normalized to:

```text
Eng-NA/Providence/Alex/011006.xml
```

The script then joins each row to `Processed_Transcripts.csv` and attaches the corresponding `transcript_id`. The notebook asserts that each normalized filename has exactly one transcript match.

The updated `child_role_with_age.csv` contains 170 rows with transcript IDs.

## 5. Correct Utterance-Level Ages for `Child` Speakers

The next step loads `All_Utterances.csv` and corrects the age assigned to utterances produced by known non-target `Child` speakers.

For each row in `child_role_with_age.csv`, the script finds utterances with the matching `transcript_id` and `speaker_code`, then replaces `target_child_age` with that child's own age in months.

After this correction, only the columns needed for downstream analysis are kept:

- `id`
- `gloss`
- `type`
- `corpus_name`
- `speaker_code`
- `speaker_role`
- `target_child_age`
- `transcript_id`

The reduced utterance table is saved as:

```text
Processed_Utterances.csv
```

After this pass, the notebook reports that 2,935,874 out of 3,194,544 utterances had filled `target_child_age` values, corresponding to 91.90% coverage. Thus, 258,670 utterances still had missing age information.

## 6. Fill Remaining Missing Ages from CHAT `@ID` Lines

Some utterances still lack `target_child_age` after the first correction. The script attempts to fill these remaining values by reading the corresponding local `.cha` files and parsing their `@ID:` lines.

For each transcript with missing utterance ages, the script builds a map from:

```text
(speaker_code, speaker_role) -> age_in_months
```

It then fills missing ages using the following logic:

1. If the utterance speaker role is not `Child` or `Target_Child`, assign the target child's age for that transcript. This gives adult/input utterances the age of the child receiving the input.
2. If the speaker role is `Child`, assign that child's own age when available. If unavailable, fall back to the target child's age.
3. If the speaker role is `Target_Child`, assign the matching target child's age when available. If unavailable, fall back to any available `Target_Child` age in the same transcript.

The output is saved as:

```text
Processed_Utterances_Updated.csv
```

This pass filled 144,187 of the 258,670 originally missing utterance ages. The notes report that the final table has 144,187 missing age values out of 3,194,544 utterances, or 4.51% missing. Therefore, approximately 95% of the utterance data can be used in age-based analyses.

## 7. Preliminary Checks Before Statistical Analysis

After creating the processed transcript and utterance files, the notebook performs several checks to guide downstream filtering and aggregation.

First, it inspects activity labels to distinguish naturalistic production from other task types. The notes identify spontaneous/naturalistic activities such as:

```text
toyplay, meal, everyday, group
```

and elicited or non-spontaneous activities such as:

```text
book, reading
```

Second, it checks the utterance `type` field in `Processed_Utterances_Updated.csv`. The observed distribution is:

| Utterance Type | Count |
| --- | ---: |
| declarative | 2,267,039 |
| question | 695,923 |
| imperative_emphatic | 135,196 |
| trail off | 60,858 |
| interruption | 13,409 |
| self interruption | 9,881 |
| quotation next line | 4,919 |
| missing CA terminator | 3,969 |
| trail off question | 1,264 |
| self interruption question | 929 |
| interruption question | 589 |
| quotation precedes | 551 |
| broken for coding | 14 |
| question exclamation | 3 |

The notebook samples examples of `trail off` utterances and defines a manual mapping from utterance type to final punctuation. This is intended to support later reconstruction or normalization of utterance strings for detector input.

Finally, the notebook checks the remaining missing-age data and reports that 670 unique `transcript_id` values still contain at least one utterance with missing `target_child_age`.

---
# Output Files

The preprocessing produces the following main files:

| File | Description |
| --- | --- |
| `Processed_Transcripts.csv` | Transcript-level table filtered to Eng-NA and annotated with `study_type` and `activity`. |
| `child_role_with_age.csv` | Non-target `Child` speakers with known ages and transcript IDs. |
| `Processed_Utterances.csv` | Reduced utterance table after correcting known non-target child speaker ages. |
| `Processed_Utterances_Updated.csv` | Final utterance table after additionally filling missing ages from local CHAT metadata. |

These files provide the basis for downstream statistical analyses of child and adult/input production by age month, corpus type, activity type, and detected filler-gap dependency construction.

---
# Notes for Reproducibility

The preprocessing assumes that the local CHAT download is available under:

```text
CHILDES-Eng-NA/
```

and that transcript filenames in `All_Transcripts.csv` follow the `childesr` convention:

```text
Eng-NA/.../*.xml
```

The workflow also assumes that `0types.txt` files are located either in the same directory as a `.cha` transcript or in one of its parent directories within the same corpus.

During cleanup, check for filename consistency in the notebook. Earlier outputs referenced variants such as `Processed_Transcripts_Updated.csv` and `Processed_Utteranes_Updated.csv`; the intended final files are `Processed_Transcripts.csv` and `Processed_Utterances_Updated.csv`.
