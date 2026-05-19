from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### SENTENCE CONSTRUCTION ###
nogap_animate_embedded = []
gap_animate_embedded = []
nogap_animate_matrix = []
gap_animate_matrix = []
nogap_inanimate_matrix = []
gap_inanimate_matrix = []
nogap_inanimate_embedded = []
gap_inanimate_embedded = []
intransitive = []
gap_subj_embedded = []
nogap_subj_embedded = []
gap_subj_matrix = []
nogap_subj_matrix = []
gap_relative = []
nogap_relative = []

# OPTION 1: Strict transitivity test sentences
nogap_animate_matrix_strict = []
gap_animate_matrix_strict = []
nogap_inanimate_matrix_strict = []
gap_inanimate_matrix_strict = []

# OPTION 4: Continuation structure test sentences (question/statement markers)
gap_animate_matrix_qmark = []
nogap_animate_matrix_period = []
gap_inanimate_matrix_qmark = []
nogap_inanimate_matrix_period = []

intransitive_verbs = ['yawned', 'fell', 'sneezed', 'arrived', 'screamed', 'cried', 'disappeared', 'laughed', 'rose', 'came']
# Removed ambitransitive: began, landed, stood, escaped, froze, jumped, tired, moved, quit, stopped
animate = ['told', 'fed', 'paid', 'promised', 'invited', 'rescued']
inanimate = ['made', 'got', 'felt', 'wanted', 'cut', 'tried', 'built', 'wrote', 'enjoyed', 'bought', 'mentioned', 'fixed', 'posted', 'realized', 'threw', 'opened', 'spread', 'ordered', 'understood', 'pulled', 'sold', 'wore', 'caught', 'changed', 'filled', 'tied', 'noticed', 'dropped', 'tasted', 'locked', 'split', 'connected', 'passed', 'recorded', 'remembered', 'loaded', 'grabbed', 'delivered', 'packed', 'cleaned']
# Strictly transitive inanimate verbs for embedded question tests only.
# Removed ambitransitive (felt, tried, changed, passed, dropped, split, connected, spread, got)
# and verbs overlapping with embedded_question_verbs (realized, understood, noticed, remembered).
inanimate_embedded = ['made', 'wanted', 'cut', 'built', 'wrote', 'enjoyed', 'bought', 'mentioned', 'fixed', 'posted', 'threw', 'opened', 'ordered', 'pulled', 'sold', 'wore', 'caught', 'filled', 'tied', 'tasted', 'locked', 'recorded', 'loaded', 'grabbed', 'delivered', 'packed', 'cleaned']
inanimate_with_objects = [['made', 'a cake'], ['got', 'the gift'], ['felt', 'the cloth'], ['wanted', 'the gift'], ['cut', 'the cake'], ['tried', 'the dish'], ['built', 'the tower'], ['wrote', 'the book'], ['enjoyed', 'the show'], ['bought', 'the gift'], ['fixed', 'the bike'], ['posted', 'the picture'], ['realized', 'the dream'], ['threw', 'the ball'], ['opened', 'the door'], ['spread', 'the news'], ['ordered', 'the package'], ['understood', 'the problem'], ['pulled', 'the rope'], ['sold', 'the book'], ['wore', 'the clothes'], ['caught', 'the ball'], ['changed', 'the color'], ['tied', 'the rope'], ['filled', 'the bucket'], ['noticed', 'the flower'], ['dropped', 'the ball'], ['tasted', 'the food'], ['locked', 'the door'], ['split', 'the apple'], ['connected', 'the dots'], ['passed', 'the test'], ['recorded', 'the song'], ['remembered', 'the song'], ['loaded', 'the car'], ['grabbed', 'the keys'], ['delivered', 'the package'], ['packed', 'the suitcase'], ['cleaned', 'the closet']]
# Removed ambitransitive verbs that are common intransitively (call, help, join, bother, teach)
# and ditransitive verbs whose nogap "bad" sentence is actually grammatical (tell: "who will he tell me" is valid).
# Added strictly transitive replacements: rescue, invite, confront, comfort, dismiss, consult.
animate_present = ['interview', 'fire', 'protect', 'thank', 'arrest', 'bless', 'scare', 'destroy', 'kiss', 'hug', 'remind', 'question', 'surprise', 'frighten', 'follow', 'blame', 'forgive', 'visit', 'greet', 'warn', 'praise', 'punish', 'trust', 'miss', 'admire', 'recognize', 'rescue', 'invite', 'confront', 'comfort', 'dismiss', 'consult']
# Removed verbs with common intransitive readings: feel, try, write, post, open, spread, order,
# understand, sell, change, fill, tie, drop, split, connect, pass, record, remember, deliver,
# pack, clean, lose, hide, break, create, design, choose, prepare, finish, collect.
# These make the gap-test "bad" sentence ("X will [verb] today") actually grammatical,
# inflating false failure rates. Added strictly transitive replacements.
inanimate_present = ['make', 'get', 'want', 'cut', 'build', 'enjoy', 'buy', 'mention', 'fix', 'realize', 'throw', 'pull', 'wear', 'catch', 'notice', 'taste', 'lock', 'load', 'grab', 'find', 'examine', 'solve', 'purchase', 'repair', 'construct', 'remove', 'replace', 'install', 'obtain', 'accomplish', 'acquire', 'inspect']

# OPTION 1: Strictly transitive verbs — subset of main lists where "*X will [verb]." (without object)
# is genuinely ungrammatical, not just degraded. Removed teach (ambitransitive+ditransitive),
# and inanimate verbs with common intransitive uses (write, post, open, order, sell, fill, tie,
# drop, pack, clean, lose, hide, break, deliver, build, throw, catch).
animate_present_strict = ['interview', 'fire', 'thank', 'arrest', 'kiss', 'hug', 'protect', 'surprise', 'frighten', 'remind', 'follow', 'blame', 'forgive', 'visit', 'rescue', 'invite', 'confront', 'comfort', 'dismiss', 'consult']
inanimate_present_strict = ['enjoy', 'mention', 'fix', 'realize', 'wear', 'notice', 'lock', 'grab', 'find', 'examine', 'solve', 'purchase', 'repair', 'construct', 'remove', 'replace', 'install', 'obtain', 'accomplish', 'acquire', 'inspect']

# OLD VERSION!!!
# embedded_question_verbs = ["wondered", "thought", "asked", "discovered", "forgot", "knew", "remembered", "saw"]
embedded_question_verbs = ["discovered", "forgot", "knew", "remembered", "saw", "noticed", "realized", "understood", "learned", "guessed"]
# Factive verbs that naturally take 'that' complements (for intransitive tests)
factive_verbs = ["knew", "realized", "noticed", "saw", "discovered", "remembered", "forgot"]
nouns = ['you', 'I', 'we', 'they', 'he', 'she', 'it', 'the doctor', 'the person', 'the singer', 'the teacher', 'the student', 'the parent', 'the child', 'the artist', 'the friend', 'John', 'Mary', 'Alex', 'the neighbor', 'Catherine', 'the astronaut']
animate_nouns = ['you', 'I', 'we', 'they', 'he', 'she', 'the doctor', 'the person', 'the singer', 'the teacher', 'the student', 'the parent', 'the child', 'the artist', 'the friend', 'John', 'Mary', 'Alex', 'the neighbor', 'Catherine', 'the astronaut']
objects = ['you', 'me', 'us', 'them', 'him', 'her', 'it', 'the doctor', 'the person', 'the singer', 'the teacher', 'the student', 'the parent', 'the child', 'the artist', 'the friend', 'John', 'Mary', 'Alex', 'the neighbor', 'Catherine', 'the astronaut']
# Object-case animate nouns only (excludes 'it') for use with animate-selecting verbs
animate_objects = ['you', 'me', 'us', 'them', 'him', 'her', 'the doctor', 'the person', 'the singer', 'the teacher', 'the student', 'the parent', 'the child', 'the artist', 'the friend', 'John', 'Mary', 'Alex', 'the neighbor', 'Catherine', 'the astronaut']

for noun in animate_nouns:
    for helper in embedded_question_verbs:
        for noun2 in animate_nouns:
            if (noun2 != noun):
                for verb in animate:
                    sentence1 = " ".join((noun, helper, "who", noun2, verb))
                    sentence2 = " ".join((noun, helper, "that", noun2, verb))
                    nogap_animate_embedded.append([sentence2, sentence1, " someone"])
                    gap_animate_embedded.append([sentence1, sentence2, " today"])

for noun in animate_nouns:
    for helper in embedded_question_verbs:
        for noun2 in animate_nouns:
            if (noun2 != noun):
                for verb in inanimate_embedded:
                    sentence1 = " ".join((noun, helper, "what", noun2, verb))
                    sentence2 = " ".join((noun, helper, "that", noun2, verb))
                    nogap_inanimate_embedded.append([sentence2, sentence1, " it"])
                    gap_inanimate_embedded.append([sentence1, sentence2, " today"])

for noun in nouns:
    for verb in animate_present:
        sentence1 = " ".join(("who will", noun, verb))
        sentence2 = " ".join((noun, "will", verb))
        nogap_animate_matrix.append([sentence2, sentence1, " someone"])
        gap_animate_matrix.append([sentence1, sentence2, " today"])

for noun in nouns:
    for verb in inanimate_present:
        sentence1 = " ".join(("what will", noun, verb))
        sentence2 = " ".join((noun, "will", verb))
        nogap_inanimate_matrix.append([sentence2, sentence1, " it"])
        gap_inanimate_matrix.append([sentence1, sentence2, " today"])

# OPTION 1: Generate matrix questions with strictly transitive verbs
for noun in nouns:
    for verb in animate_present_strict:
        sentence1 = " ".join(("who will", noun, verb))
        sentence2 = " ".join((noun, "will", verb))
        nogap_animate_matrix_strict.append([sentence2, sentence1, " someone"])
        gap_animate_matrix_strict.append([sentence1, sentence2, " today"])

for noun in nouns:
    for verb in inanimate_present_strict:
        sentence1 = " ".join(("what will", noun, verb))
        sentence2 = " ".join((noun, "will", verb))
        nogap_inanimate_matrix_strict.append([sentence2, sentence1, " it"])
        gap_inanimate_matrix_strict.append([sentence1, sentence2, " today"])

# OPTION 4: Generate matrix questions with punctuation mark continuations
# Both prompts get same continuation; only the prompt differs
for noun in nouns:
    for verb in animate_present:
        # Gap test: "who will John call?" (good) vs "John will call?" (bad, missing object)
        sentence1 = " ".join(("who will", noun, verb))
        sentence2 = " ".join((noun, "will", verb))
        gap_animate_matrix_qmark.append([sentence1, sentence2, "?"])
        
        # Nogap test: "John will call someone." (good) vs "who will John call someone." (bad, double object)
        sentence1_complete = " ".join((noun, "will", verb, "someone"))
        sentence2_double = " ".join(("who will", noun, verb, "someone"))
        nogap_animate_matrix_period.append([sentence1_complete, sentence2_double, "."])

for noun in nouns:
    for verb in inanimate_present:
        # Gap test: "what will John build?" (good) vs "John will build?" (bad, missing object)
        sentence1 = " ".join(("what will", noun, verb))
        sentence2 = " ".join((noun, "will", verb))
        gap_inanimate_matrix_qmark.append([sentence1, sentence2, "?"])
        
        # Nogap test: "John will build something." (good) vs "what will John build something." (bad, double object)
        sentence1_complete = " ".join((noun, "will", verb, "something"))
        sentence2_double = " ".join(("what will", noun, verb, "something"))
        nogap_inanimate_matrix_period.append([sentence1_complete, sentence2_double, "."])


for noun in animate_nouns:
    for helper in factive_verbs:
        for noun2 in animate_nouns:
            if (noun2 != noun):
                for verb in intransitive_verbs:
                    sentence1 = " ".join((noun, helper, "what", noun2, verb))
                    sentence2 = " ".join((noun, helper, "that", noun2, verb))
                    intransitive.append([sentence2, sentence1, " today"])

for noun in animate_nouns:
    for helper in embedded_question_verbs:
        for noun2 in animate_objects:
            if (noun2 != noun):
                for verb in animate:
                    sentence1 = " ".join((noun, helper, "who"))
                    sentence2 = " ".join((noun, helper, "that"))
                    sentence3 = " " + " ".join((verb, noun2))
                    gap_subj_embedded.append([sentence1, sentence2, sentence3])
                    sentence1 = " ".join((noun, helper, "who", 'they'))
                    sentence2 = " ".join((noun, helper, "that", "they"))
                    sentence3 = " " + " ".join((verb, noun2))
                    nogap_subj_embedded.append([sentence2, sentence1, sentence3])

for noun2 in objects:
    for verb in animate_present:
        # Gap: "who will call me" (good) vs "will call me" (bad, missing subject)
        sentence1 = "who will"
        sentence2 = "will"
        sentence3 = " " + " ".join((verb, noun2))
        gap_subj_matrix.append([sentence1, sentence2, sentence3])

# Vary the subject pronoun to avoid confounding with a single lexical item.
# Removed ditransitive verbs (tell, teach) from animate_present to prevent
# the "bad" sentence from being grammatical (e.g., "who will he tell me" is valid ditransitive).
subj_matrix_nouns = ['he', 'she', 'they']
for subj in subj_matrix_nouns:
    for noun2 in objects:
        if subj != noun2:
            for verb in animate_present:
                # Nogap: "he will rescue me" (good) vs "who will he rescue me" (bad, extra wh-word)
                sentence1 = subj + " will"
                sentence2 = "who will " + subj
                sentence3 = " " + " ".join((verb, noun2))
                nogap_subj_matrix.append([sentence1, sentence2, sentence3])
        
# Relative clause tests: gap vs resumptive pronoun
# Gap: "I saw a cake that you made yesterday" (good) vs "I saw a cake that you made it yesterday" (bad, resumptive)
# Nogap: "I knew that you made a cake yesterday" (good) vs "I knew that you made a cake it yesterday" (bad, extra pronoun)
# Multiple matrix/complement verbs to avoid confounding gap knowledge with verb-specific preferences.
rc_matrix_verbs = ['saw', 'found', 'liked']
rc_complement_verbs = ['knew', 'realized', 'believed']
for noun in animate_nouns:
    for noun2 in animate_nouns:
        if noun != noun2:
            for pair in inanimate_with_objects:
                head_noun = pair[1]  # e.g., "a cake"
                verb = pair[0]       # e.g., "made"
                # Gap test: relative clause with gap vs resumptive pronoun
                # good: "I saw a cake that you made yesterday"
                # bad:  "I saw a cake that you made it yesterday"
                for rc_verb in rc_matrix_verbs:
                    sentence_good = " ".join((noun, rc_verb, head_noun, "that", noun2, verb))
                    sentence_bad = " ".join((noun, rc_verb, head_noun, "that", noun2, verb, "it"))
                    gap_relative.append([sentence_good, sentence_bad, " yesterday"])
                # Nogap test: that-clause with proper object vs doubled object
                # good: "I knew that you made a cake yesterday"
                # bad:  "I knew that you made a cake it yesterday"
                for comp_verb in rc_complement_verbs:
                    sentence_good = " ".join((noun, comp_verb, "that", noun2, verb, head_noun))
                    sentence_bad = " ".join((noun, comp_verb, "that", noun2, verb, head_noun, "it"))
                    nogap_relative.append([sentence_good, sentence_bad, " yesterday"])

sentence_dict = {
        "nogap_animate_embedded": nogap_animate_embedded,
        "gap_animate_embedded": gap_animate_embedded,
        "nogap_animate_matrix": nogap_animate_matrix,
        "gap_animate_matrix": gap_animate_matrix,
        "nogap_inanimate_matrix": nogap_inanimate_matrix,
        "gap_inanimate_matrix": gap_inanimate_matrix,
        "nogap_inanimate_embedded": nogap_inanimate_embedded,
        "gap_inanimate_embedded": gap_inanimate_embedded,
        "intransitive": intransitive,
        "gap_subj_embedded": gap_subj_embedded,
        "nogap_subj_embedded": nogap_subj_embedded,
        "gap_subj_matrix": gap_subj_matrix,
        "nogap_subj_matrix": nogap_subj_matrix,
        "gap_relative": gap_relative,
        "nogap_relative": nogap_relative,
        "animate_embedded": gap_animate_embedded + nogap_animate_embedded,
        "inanimate_embedded": gap_inanimate_embedded + nogap_inanimate_embedded,
        "animate_matrix": gap_animate_matrix + nogap_animate_matrix,
        "inanimate_matrix": gap_inanimate_matrix + nogap_inanimate_matrix,
        "subj_embedded": gap_subj_embedded + nogap_subj_embedded,
        "subj_matrix": gap_subj_matrix + nogap_subj_matrix,
        "embedded": gap_animate_embedded + gap_inanimate_embedded + gap_subj_embedded + nogap_animate_embedded + nogap_inanimate_embedded + nogap_subj_embedded,
        "matrix": gap_animate_matrix + gap_inanimate_matrix + gap_subj_matrix + nogap_animate_matrix + nogap_inanimate_matrix + nogap_subj_matrix,
        "relative": gap_relative + nogap_relative,
        # OPTION 1: Strict transitivity tests
        "animate_matrix_strict": gap_animate_matrix_strict + nogap_animate_matrix_strict,
        "inanimate_matrix_strict": gap_inanimate_matrix_strict + nogap_inanimate_matrix_strict,
        "matrix_strict": gap_animate_matrix_strict + gap_inanimate_matrix_strict + nogap_animate_matrix_strict + nogap_inanimate_matrix_strict,
        # OPTION 4: Continuation marker tests
        "animate_matrix_qmark": gap_animate_matrix_qmark,
        "animate_matrix_period": nogap_animate_matrix_period,
        "inanimate_matrix_qmark": gap_inanimate_matrix_qmark,
        "inanimate_matrix_period": nogap_inanimate_matrix_period,
        "matrix_continuation": gap_animate_matrix_qmark + nogap_animate_matrix_period + gap_inanimate_matrix_qmark + nogap_inanimate_matrix_period,
    }

def evaluate_log_prob(prompt, continuation, tokenizer, model):
    total_sentence = prompt + continuation
    encoding = tokenizer.encode(total_sentence, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_tokens = tokenizer(prompt, add_special_tokens = False, return_tensors="pt").to(device)
    prompt_len = prompt_tokens.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(encoding)
        logits = outputs.logits
    relevant_logits = logits[:, prompt_len - 1 : -1, :]
    relevant_labels = encoding[:, prompt_len:]

    # Reshape for cross_entropy
    relevant_logits = relevant_logits.reshape(-1, relevant_logits.size(-1))
    relevant_labels = relevant_labels.reshape(-1)
    nll = F.cross_entropy(relevant_logits, relevant_labels, reduction='sum')
    log_likelihood = -nll.item()
    return log_likelihood

def evaluate(all_sentences, tokenizer, model):
    count = 0
    correct = 0
    for sentences in tqdm.tqdm(all_sentences):

        prob1 = evaluate_log_prob(sentences[0], sentences[2], tokenizer, model)
        prob2 = evaluate_log_prob(sentences[1], sentences[2], tokenizer, model)

        if (prob1 > prob2):
            correct += 1
        count += 1
    return (correct/count)

def analyze_final(dataset_filter, model_config, dataset_size, include_dir, seed, minimal_pair_category, lr=3e-4, wd=0.0, control=False, model_root=None):
    base_model_path = "./models"

    model_paths_and_labels = []
    
    # Format lr and wd for directory name (e.g., lr3e04_wd0e+00)
    # Must match train.py: both strip hyphens from scientific notation
    lr_str = f"{lr:.0e}".replace('-', '')
    wd_str = f"{wd:.0e}".replace('-', '')
    lr_wd_dir = f"lr{lr_str}_wd{wd_str}"
    
    if model_root:
        # Balanced-membership dataset mode: direct model path
        model_root_path = Path(model_root)
        model_path = model_root_path / f"{model_config}-{dataset_size}" / lr_wd_dir / f"seed_{seed}" / "final"
        model_label = f"{dataset_filter}-{include_dir}-{model_config}-{dataset_size}-{seed}-{lr_wd_dir}"
    elif control:
        model_path = Path(base_model_path, include_dir, dataset_filter, f"{model_config}-{dataset_size}-control", lr_wd_dir, f"seed_{seed}", "final")
        model_label = f"{dataset_filter}-{include_dir}-{model_config}-{dataset_size}-{seed}-{lr_wd_dir}-control"
    else:
        model_path = Path(base_model_path, include_dir, dataset_filter, f"{model_config}-{dataset_size}", lr_wd_dir, f"seed_{seed}", "final")
        model_label = f"{dataset_filter}-{include_dir}-{model_config}-{dataset_size}-{seed}-{lr_wd_dir}"

    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
        return

    # Validate minimal_pair_category
    if minimal_pair_category not in sentence_dict:
        print(f"Invalid minimal pair category: {minimal_pair_category}")
        print(f"Valid categories: {list(sentence_dict.keys())}")
        return

    model_paths_and_labels.append((model_path, model_label))

    for pair in model_paths_and_labels:
        model_path, model_label = pair
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        # Evaluate only the specified minimal pair category
        sentence_list = sentence_dict[minimal_pair_category]
        
        # Sample 5000 minimal pairs if category has more than 5000
        if len(sentence_list) > 5000:
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(len(sentence_list), size=5000, replace=False)
            sentence_list = [sentence_list[i] for i in indices]
            print(f"Sampled 5000 minimal pairs from {len(sentence_dict[minimal_pair_category])} total pairs")
        
        proportion_right = evaluate(sentence_list, tokenizer, model)
        print(f"{model_label} on {minimal_pair_category}: {proportion_right}")
    
        # Save result to CSV
        columns = ["model", "dataset_filter", "model_config", "dataset_size", "include_dir", "seed", "lr", "wd", "control", "minimal_pair_category", "proportion_right"]
        data = [(model_label, dataset_filter, model_config, dataset_size, include_dir, seed, lr, wd, control, minimal_pair_category, proportion_right)]
        df = pd.DataFrame(data, columns=columns)
        
        # Create results directory structure
        if model_root:
            # For balanced datasets, mirror the model_root structure in analysis_data_balanced
            results_dir = Path("analysis_data_balanced", include_dir, dataset_filter, f"{model_config}-{dataset_size}", lr_wd_dir, f"seed_{seed}")
        else:
            results_dir = Path("analysis_data", include_dir, dataset_filter, f"{model_config}-{dataset_size}{'-control' if control else ''}", lr_wd_dir, f"seed_{seed}")
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"{minimal_pair_category}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

def analyze_checkpoints(dataset_filter, model_config, dataset_size, include_dir, seed, minimal_pair_category, lr=3e-4, wd=0.0, control=False):
    base_model_path = "./models"

    model_paths_and_labels = []
    checkpoints = []

    # Format lr and wd for directory name (e.g., lr3e04_wd0e+00)
    lr_str = f"{lr:.0e}".replace('-', '')
    wd_str = f"{wd:.0e}".replace('-', '')
    lr_wd_dir = f"lr{lr_str}_wd{wd_str}"

    if control:
        dir_path = Path(base_model_path, include_dir, dataset_filter, f"{model_config}-{dataset_size}-control", lr_wd_dir, f"seed_{seed}")
        label_suffix = "-control"
    else:
        dir_path = Path(base_model_path, include_dir, dataset_filter, f"{model_config}-{dataset_size}", lr_wd_dir, f"seed_{seed}")
        label_suffix = ""

    if not dir_path.exists():
        print(f"Directory path does not exist: {dir_path}")
        return

    # Validate minimal_pair_category
    if minimal_pair_category not in sentence_dict:
        print(f"Invalid minimal pair category: {minimal_pair_category}")
        print(f"Valid categories: {list(sentence_dict.keys())}")
        return

    for entry in dir_path.iterdir():
        if entry.is_dir() and "checkpoint" in entry.name:
            checkpoints.append(entry.name)

    checkpoints.sort()

    for checkpoint in checkpoints:
        model_label = f"{dataset_filter}-{include_dir}-{model_config}-{dataset_size}-{seed}-{checkpoint}{label_suffix}"
        model_path = dir_path / checkpoint

        if model_path.exists():
            model_paths_and_labels.append((model_path, model_label, checkpoint))
    
    for pair in model_paths_and_labels:
        model_path, model_label, checkpoint = pair
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        # Evaluate only the specified minimal pair category
        sentence_list = sentence_dict[minimal_pair_category]
        
        # Sample 5000 minimal pairs if category has more than 5000
        if len(sentence_list) > 5000:
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(len(sentence_list), size=5000, replace=False)
            sentence_list = [sentence_list[i] for i in indices]
            print(f"Sampled 5000 minimal pairs from {len(sentence_dict[minimal_pair_category])} total pairs")
        
        proportion_right = evaluate(sentence_list, tokenizer, model)
        print(f"{model_label} on {minimal_pair_category}: {proportion_right}")
    
        # Extract checkpoint number from model_label
        checkpoint_num = checkpoint.split('-')[-1]  # e.g., "checkpoint-1000" -> "1000"
        
        # Save result to CSV
        columns = ["model", "dataset_filter", "model_config", "dataset_size", "include_dir", "seed", "checkpoint", "lr", "wd", "control", "minimal_pair_category", "proportion_right"]
        data = [(model_label, dataset_filter, model_config, dataset_size, include_dir, seed, checkpoint_num, lr, wd, control, minimal_pair_category, proportion_right)]
        df = pd.DataFrame(data, columns=columns)
        
        # Create results directory structure
        results_dir = Path("analysis_data", include_dir, dataset_filter, f"{model_config}-{dataset_size}{'-control' if control else ''}", lr_wd_dir, f"seed_{seed}", checkpoint)
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / f"{minimal_pair_category}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

def evaluation_unit_test():
    model_path = Path("models", "par", "embedded_questions", "llama-360M-10M", "seed_0", "final")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    print(evaluate_log_prob("he ate the fruit that fell from the tree", "fast", tokenizer, model))
    print(evaluate_log_prob("he ate the fruit that fell", "fast", tokenizer, model))

def parse_args():
    parser = argparse.ArgumentParser(description="Run linguistic analysis on trained models")
    parser.add_argument("--dataset_filter", type=str, required=True,
                        help="Dataset filter type (e.g., embedded_questions, matrix_questions, relative_clauses, or coalition name like embq+matq)")
    parser.add_argument("--model_config", type=str, required=True,
                        help="Model configuration (e.g., llama-360M, gpt-705M)")
    parser.add_argument("--dataset_size", type=str, required=True,
                        choices=["10M", "100M"],
                        help="Dataset size")
    parser.add_argument("--include_dir", type=str, required=True,
                        help="Include directory (e.g., par, all)")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed (0-4)")
    parser.add_argument("--minimal_pair_category", type=str, required=True,
                        help="Minimal pair category to evaluate")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate used during training (default: 3e-4)")
    parser.add_argument("--wd", type=float, default=0.0,
                        help="Weight decay used during training (default: 0.0)")
    parser.add_argument("--eval_checkpoints", action="store_true",
                        help="Whether to evaluate checkpoints (default: False)")
    parser.add_argument("--control", action="store_true",
                        help="Evaluate control model instead of filtered model (default: False)")
    # Balanced-membership dataset support
    parser.add_argument("--model_root", type=str, default=None,
                        help="Direct path to model root dir (e.g. models_balanced/par/rep_0/embq+matq). "
                             "Bypasses filter-based model path construction.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_type = "control" if args.control else "filtered"
    print(f"Running analysis for: {args.dataset_filter}, {args.model_config}, {args.dataset_size}, {args.include_dir}, seed {args.seed}, lr={args.lr}, wd={args.wd}, model type: {model_type}, minimal pair: {args.minimal_pair_category}")
    if args.eval_checkpoints:
        print("Evaluating checkpoints...")
        analyze_checkpoints(args.dataset_filter, args.model_config, args.dataset_size, args.include_dir, args.seed, args.minimal_pair_category, args.lr, args.wd, args.control)
    analyze_final(args.dataset_filter, args.model_config, args.dataset_size, args.include_dir, args.seed, args.minimal_pair_category, args.lr, args.wd, args.control, model_root=args.model_root)
    # evaluation_unit_test()
