import json
import sys
import spacy
import benepar
from tqdm import tqdm
from nltk.tree import Tree

from src.MatrixQ_Detector import MatrixQuestionDetector
from src.EmbQ_Detector import EmbeddedQuestionDetector
from src.RC_Detector import RelativeClauseDetector 
import re
########################################################################
# Parse arguments
########################################################################
dataset = sys.argv[1] # dev, test, train_100M, train_10M
child_filtered = False if sys.argv[2].lower() == 'false' else bool(sys.argv[2])
start_idx = int(sys.argv[3])

########################################################################
# Get dataset
########################################################################
def remove_brackets(sentence: str) -> str:
    # Remove [ ... ] including the brackets
    cleaned = re.sub(r"\[.*?\]", "", sentence)
    # Collapse extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Fix stray spaces before punctuation
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
    return cleaned

suffix = dataset.split('_')[0]
file_children = 'noChild' if child_filtered else 'Child'
file_path = f'BabyLM/text_data/{dataset}/childes.{suffix}'
output_path = f'all_labeled_data/fragment_files/{dataset}_{file_children}_{start_idx}.jsonl'

print(f'File path = {file_path}')
print(f'Output path = {output_path}')

with open(file_path, 'r') as f:
    raw = f.readlines()
conversations = [r for r in raw if r[0] == "*"]
if child_filtered:
    utterances = ["\t".join(c.split("\t")[1:]).strip() for c in conversations if not c.startswith("*CHI:")]
else:
    utterances = ["\t".join(c.split("\t")[1:]).strip() for c in conversations if c.startswith("*CHI:")]
print(f"Total number of sentences: {len(utterances)}")

########################################################################
# Load models and detectors
########################################################################
eq_detector = EmbeddedQuestionDetector()
rc_detector = RelativeClauseDetector()
mq_detector = MatrixQuestionDetector()
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
benepar_parser = benepar.Parser("benepar_en3_large")

def get_constituency_parse(sentence: str) -> Tree:
    """Get constituency parse tree using benepar."""
    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    return benepar_parser.parse(sent.text)

def get_dependency_parse(sentence: str):
    """Get dependency parse using spaCy."""
    return nlp(sentence)

########################################################################
# Get dataset
########################################################################
if dataset != 'train_100M':
    BS = len(utterances) if len(utterances) < 200000 else int(len(utterances) / 2)
else:
    BS = 186530
for sen in tqdm(utterances[start_idx * BS: min((start_idx + 1) * BS, len(utterances))]):
    sen_clean = remove_brackets(sen)
    result = {}
    try:
        tree = get_constituency_parse(sen_clean)
        dep = get_dependency_parse(sen_clean)
    except Exception:
        continue
    result['sentence'] = sen
    result['sentence_clean'] = sen_clean
    result['speaker'] = 'CHI' if not child_filtered else 'PAR' 
    result['MatrixQ'] = mq_detector.detect_matrix_questions(sen_clean, tree, dep)
    result['EmbQ'] = eq_detector.detect_embedded_questions(sen_clean, tree, dep)
    result['RC'] = rc_detector.detect_relative_clauses(sen_clean, tree, dep)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result) + '\n')