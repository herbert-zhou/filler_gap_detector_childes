install.packages("childesr")
library(childesr)
library(dplyr)
library(data.table)
library(readr)
library(jsonlite)

d_eng_na <- get_transcripts(collection = "Eng-NA")
head(d_eng_na)
# check d_eng_na structure
str(d_eng_na)

# # returns all transcripts in the brown corpus
# d_brown_transcripts <- get_transcripts(corpus = "Brown")
# # print the number of rows
# nrow(d_brown_transcripts)

d_adam_utts <- get_utterances(corpus = "Brown",
                              target_child = "Adam")

# view the structure of the data
str(d_adam_utts)


# getting all utterances
utts <- get_utterances(collection = "Eng-NA", language = "eng")
# save utts to csv
utts_clean <- utts %>%
    # convert list-columns to JSON strings (if any)
    mutate(across(where(is.list), ~ toJSON(.x, auto_unbox = TRUE))) %>%
    # optional: normalize factors (if any) to plain strings
    mutate(across(where(is.factor), as.character))
# write a standards-friendly CSV
fwrite(
    utts_clean,
    file = "All_Utterances.csv",
    na = "", # <-- blanks instead of "NA"
    bom = TRUE # helpful for Excel; safe otherwise
)

# getting all transcripts information
transcripts <- get_transcripts(  collection = NULL,  corpus = NULL,  target_child = NULL,  connection = NULL,  db_version = "current",  db_args = NULL)
# save to csv
transcripts_clean <- transcripts %>%
    # convert list-columns to JSON strings (if any)
    mutate(across(where(is.list), ~ toJSON(.x, auto_unbox = TRUE))) %>%
    # optional: normalize factors (if any) to plain strings
    mutate(across(where(is.factor), as.character))
# write a standards-friendly CSV
fwrite(
    transcripts_clean,
    file = "All_Transcripts.csv",
    na = "", # <-- blanks instead of "NA"
    bom = TRUE # helpful for Excel; safe otherwise
)



# for column "type", count the number of occurrences of each unique value
counts <- utts %>%
  group_by(speaker_role) %>%
  summarise(count = n())
# sort counts in descending order
counts <- counts %>%
  arrange(desc(count))
print(counts, n=nrow(counts))

# for all utterances with speaker_role = "Target_Child" or "Child", see whether the "target_child_age" column has any missing values
child_utts <- utts %>%
  filter(speaker_role %in% c("Target_Child", "Child"))
missing_ages <- child_utts %>%
  filter(is.na(target_child_age))
# print the number of missing values
print(nrow(missing_ages) / nrow(child_utts))

# for all utterances with speaker_role NOT EQUAL TO "Target_Child" or "Child", check how many of them have target_child_age
other_utts <- utts %>%
  filter(!speaker_role %in% c("Target_Child", "Child"))
other_missing_ages <- other_utts %>%
  filter(is.na(target_child_age))
# print the number of missing values
print(nrow(other_missing_ages) / nrow(other_utts))

other_utts <- utts %>%
  filter(speaker_role %in% c("Sister"))
other_missing_ages <- other_utts %>%
  filter(!is.na(target_child_age))
# print the number of missing values
print(nrow(other_missing_ages) / nrow(other_utts))

# for speaker role "Brother", check how many of them have target_child_age
other_utts <- utts %>%
  filter(!speaker_role %in% c("Target_Child", 'Child'))
other_with_ages <- other_utts %>%
  filter(!is.na(target_child_age))
# print the number of missing values
print(nrow(other_with_ages) / nrow(other_utts))
# print average ages
print(mean(other_with_ages$target_child_age, na.rm = TRUE))
