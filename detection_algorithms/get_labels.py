import spacy
import benepar
from nltk.tree import Tree

from detection_algorithms.MatrixQ_Detector import MatrixQuestionDetector
from detection_algorithms.EmbQ_Detector import EmbeddedQuestionDetector
from detection_algorithms.RC_Detector import RelativeClauseDetector 

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
# Helpers for aggregating after detection 
########################################################################
Permitted_EmbQ_Lemma = ['know', 'see', 'tell', 'look', 'remember', 'wonder', 'guess',
                        'ask', 'say', 'forget', 'figure', 'understand', 'decide', 
                        'show', 'watch', 'hear', 'think']
xEver = ['whatever', 'whoever', 'whomever', 'whichever', 'whenever', 'wherever']

EmbQ_Convert = {'subject': 'SEQ', 'object': 'OEQ', 'adjunct': 'AEQ', 'polar': 'PEQ'}

ALL_TYPES = {'MatrixQ': ['SMQ', 'OMQ', 'AMQ', 'PMQ', 'PlainMQ', 'CC_SMQ', 'CC_OMQ', 'CC_AMQ'],
             'EmbQ': ['SEQ', 'OEQ', 'AEQ', 'PEQ'],
             'RC': ['SRC', 'ORC', 'ARC', 'PRC', 'SRC_reduced', 'ORC_reduced']}

def check_MatrixQ(mq_label):
    '''Return a (potentially empty) list of all matrix question types in the sentence.'''
    if mq_label is not None:
        return [mq_label]
    return []

def check_EmbQ(eq_labels, sen):
    '''Return a (potentially empty) list of all embedded question gap types in the sentence.'''
    if len(eq_labels) > 0: # if there is at least one embedded question
        if not any(word.lower() in sen.lower() for word in xEver):  # ignore all ~ever case
            labels = []
            for embq in eq_labels: # iterate through all embedded questions
                if embq['gap_type'] != 'unknown' and embq['selecting_verb'] is not None: # only consider known gap types with a selecting verb
                    if nlp(embq['selecting_verb'])[0].lemma_ in Permitted_EmbQ_Lemma: # only consider selected verb lemmas
                        labels.append(EmbQ_Convert[embq['gap_type']]) # convert and add to labels
            return labels

def check_RC(rc_labels):
    '''Return a (potentially empty) list of all relative clause gap types in the sentence.'''
    if len(rc_labels) > 0:
        labels = []
        for item in rc_labels:
            if item['gap_type'] in ['SRC', 'ORC', 'ARC', 'PRC'] and item['wh_word'] is not None:
                labels.append(item['gap_type'])
            elif item['gap_type'] in ['SRC_reduced', 'ORC_reduced']:
                labels.append(item['gap_type'])
            else:
                continue
        return labels

    
########################################################################
# One sentence processing, end-to-end
########################################################################    
def get_labels(sen):
    labels = []
    try:
        tree = get_constituency_parse(sen)
        dep = get_dependency_parse(sen)
    except Exception:
        return ['ParsingError']
    
    labels.extend(check_MatrixQ(mq_detector.detect_matrix_questions(sen, tree, dep)) or [])
    labels.extend(check_EmbQ(eq_detector.detect_embedded_questions(sen, tree, dep), sen) or [])
    labels.extend(check_RC(rc_detector.detect_relative_clauses(sen, tree, dep)) or [])
    
    return labels