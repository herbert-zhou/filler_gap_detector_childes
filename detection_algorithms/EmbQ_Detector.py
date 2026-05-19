import csv
import json
import sys
import spacy
import benepar
from tqdm import tqdm
from nltk.tree import Tree
from typing import List, Tuple, Optional, Dict, Set
import re

# Hybrid approach: use Constituency parse to identify the VP-SBAR struture, and use dependency parse for gap type classification

class EmbeddedQuestionDetector:
    def __init__(self):
        # spacy.prefer_gpu()
        # self.nlp = spacy.load("en_core_web_trf")
        # self.benepar_parser = benepar.Parser("benepar_en3_large")
        
        # WH-words and question markers
        self.wh_words = {
            'what', 'who', 'whom', 'whose', 'which', 'when', 'where', 'why', 'how',
            'whatever', 'whoever', 'whomever', 'whenever', 'wherever', 'however'
        }
        self.question_markers = {'if', 'whether'}
        
    # def get_constituency_parse(self, sentence: str) -> Tree:
    #     """Get constituency parse tree using benepar."""
    #     doc = self.nlp(sentence)
    #     sent = list(doc.sents)[0]
    #     return self.benepar_parser.parse(sent.text)
    
    # def get_dependency_parse(self, sentence: str):
    #     """Get dependency parse using spaCy."""
    #     return self.nlp(sentence)
    
    def find_vp_sbar_structures(self, tree: Tree) -> List[Dict]:
        """Find VP-SBAR structures where VP immediately dominates SBAR."""
        vp_sbar_structures = []
        
        def traverse(node, path=[]):
            if isinstance(node, Tree):
                current_path = path + [node.label()]
                
                # Check if this is a VP that immediately dominates SBAR
                if node.label() == 'VP':
                    for child in node:
                        if isinstance(child, Tree) and child.label() == 'SBAR':
                            sbar_info = self.analyze_sbar(child, current_path)
                            if sbar_info:  # Only add if it's a valid embedded question
                                # Find selecting verb from this VP
                                selecting_verb = self.extract_verb_from_vp(node)
                                sbar_info['selecting_verb'] = selecting_verb
                                vp_sbar_structures.append(sbar_info)
                
                # Recursively check children
                for child in node:
                    traverse(child, current_path)
        
        traverse(tree)
        return vp_sbar_structures
    
    def analyze_sbar(self, sbar_node: Tree, vp_path: List[str]) -> Optional[Dict]:
        """Analyze SBAR to determine if it's an embedded question."""
        sbar_info = {
            'sbar_node': sbar_node,
            'text': ' '.join(sbar_node.leaves()),
            'has_wh': False,
            'has_question_marker': False,
            'first_word': None,
            'first_child_label': None,
            'is_that_clause': False,
            'vp_path': vp_path
        }
        
        # Analyze first child
        if len(sbar_node) > 0:
            first_child = sbar_node[0]
            if isinstance(first_child, Tree):
                sbar_info['first_child_label'] = first_child.label()
                
                if first_child.label().startswith('WH'):
                    sbar_info['has_wh'] = True
                    sbar_info['first_word'] = ' '.join(first_child.leaves())
                elif first_child.label() == 'IN':
                    first_word = ' '.join(first_child.leaves()).lower()
                    sbar_info['first_word'] = first_word
                    
                    if first_word in self.question_markers:
                        sbar_info['has_question_marker'] = True
                    elif first_word == 'that':
                        sbar_info['is_that_clause'] = True
                        return None  # Filter out "that" clauses
        
        # Only return if it's a valid embedded question
        if sbar_info['has_wh'] or sbar_info['has_question_marker']:
            return sbar_info
        
        return None
    
    def extract_verb_from_vp(self, vp_node: Tree) -> Optional[str]:
        """Extract the main verb from a VP node."""
        if not isinstance(vp_node, Tree):
            return None
        
        def find_main_verb(node):
            if isinstance(node, Tree):
                # Check if this is a verb node (prioritize main verbs)
                if node.label().startswith('V') and node.label() != 'VP':
                    leaves = node.leaves()
                    if leaves:
                        return leaves[0]
                
                # Search children for verbs
                for child in node:
                    if isinstance(child, Tree) and child.label().startswith('V') and child.label() != 'VP':
                        leaves = child.leaves()
                        if leaves:
                            return leaves[0]
                
                # Deeper search if no direct verb children
                for child in node:
                    result = find_main_verb(child)
                    if result:
                        return result
            
            return None
        
        verb = find_main_verb(vp_node)
        return verb.lower() if verb else None
    
    def find_wh_words_dependency(self, doc) -> List[Dict]:
        """Find WH-words using dependency parsing."""
        wh_tokens = []
        for token in doc:
            if (token.pos_ in ['WP', 'WDT', 'WRB'] or 
                token.text.lower() in self.wh_words or 
                token.text.lower() in self.question_markers):
                wh_tokens.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'dep': token.dep_,
                    'head': token.head.text,
                    'head_pos': token.head.pos_,
                    'head_dep': token.head.dep_,
                    'token': token
                })
        return wh_tokens
    
    def analyze_gap_type_hybrid(self, sbar_info: Dict, doc, sentence: str) -> str:
        """
        Analyze gap type using hybrid approach with corrected dependency classification.
        """
        if not sbar_info['first_word'] or sbar_info['first_word'].lower() in self.question_markers:
            return 'polar'
        
        # Get WH-word tokens from dependency parse
        wh_tokens = self.find_wh_words_dependency(doc)
        
        # Find the relevant WH-token
        relevant_wh_token = None
        wh_actual = self.extract_wh_word(sbar_info['first_word'])
        
        for wh_token in wh_tokens:
            if wh_token['text'].lower() == wh_actual or wh_token['lemma'].lower() == wh_actual:
                relevant_wh_token = wh_token
                break
        
        if not relevant_wh_token:
            return self.analyze_gap_type_constituency_fallback(sbar_info)
        
        # Corrected dependency-based classification
        dep_type = relevant_wh_token['dep']
        
        # Primary classification based on dependency relations
        if dep_type in ['nsubj', 'nsubjpass', 'csubj']:
            return 'subject'
        elif dep_type in ['dobj', 'iobj', 'attr', 'oprd', 'dative']:
            return 'object'
        elif dep_type in ['pobj', 'advmod', 'prep', 'npadvmod', 'tmod']:
            return 'adjunct'
        elif dep_type == 'det':
            # Special case: check if WH structure is WHPP
            if self.is_whpp_structure(sbar_info['sbar_node']):
                return 'adjunct'
            else:
                return 'object'
        elif dep_type == 'ROOT':
            return 'subject'
        else:
            # Use additional heuristics for unclear cases
            return self.classify_unclear_dependency(relevant_wh_token, sbar_info)
    
    def is_whpp_structure(self, sbar_node: Tree) -> bool:
        """Check if the WH structure under SBAR is WHPP (WH prepositional phrase)."""
        if not isinstance(sbar_node, Tree):
            return False
        
        # Look for WHPP as first child of SBAR
        for child in sbar_node:
            if isinstance(child, Tree) and child.label() == 'WHPP':
                return True
        
        return False
    
    def classify_unclear_dependency(self, wh_token: Dict, sbar_info: Dict) -> str:
        """Handle unclear dependency cases with additional heuristics."""
        wh_text = wh_token['text'].lower()
        head_pos = wh_token['head_pos']
        
        # Heuristic 1: WH-word type
        adjunct_wh_words = {'when', 'where', 'why', 'how'}
        
        if wh_text in adjunct_wh_words:
            return 'adjunct'
        
        # Heuristic 2: Head POS tag
        if head_pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            # For verb-dependent WH-words, default to object unless clearly adjunct
            if wh_text in adjunct_wh_words:
                return 'adjunct'
            else:
                return 'object'
        
        # Fallback to constituency analysis
        return self.analyze_gap_type_constituency_fallback(sbar_info)
    
    def analyze_gap_type_constituency_fallback(self, sbar_info: Dict) -> str:
        """Simplified constituency fallback analysis."""
        sbar_node = sbar_info['sbar_node']
        wh_word = sbar_info['first_word']
        
        wh_actual = self.extract_wh_word(wh_word)
        if not wh_actual:
            return 'unknown'
        
        adjunct_wh_words = {'when', 'where', 'why', 'how'}
        
        try:
            # Find S node in SBAR
            s_node = None
            for child in sbar_node:
                if isinstance(child, Tree) and child.label() == 'S':
                    s_node = child
                    break
            
            if not s_node:
                return 'unknown'
            
            # Simple constituency heuristic
            s_children_labels = [child.label() if isinstance(child, Tree) else 'LEAF' 
                               for child in s_node]
            
            if 'VP' in s_children_labels:
                # Check for overt subject NP before VP
                has_np_subject = False
                vp_index = s_children_labels.index('VP')
                
                for i in range(vp_index):
                    if s_children_labels[i] == 'NP':
                        has_np_subject = True
                        break
                
                if not has_np_subject:
                    return 'subject'
                elif wh_actual in adjunct_wh_words:
                    return 'adjunct'
                else:
                    return 'object'
            
            return 'unknown'
            
        except Exception:
            return 'unknown'
    
    def extract_wh_word(self, wh_phrase: str) -> str:
        """Extract the actual WH-word from a WH-phrase."""
        if not wh_phrase:
            return None
        
        words = wh_phrase.strip().lower().split()
        
        for word in words:
            if word in self.wh_words:
                return word
        
        return words[0] if words else None


    def detect_embedded_questions(self, sentence: str, constituency_tree, dependency_doc) -> List[Dict]:
        """Main function to detect embedded questions in a sentence. Returns simplified structure."""
        try:
            # Get both parses
            # constituency_tree = self.get_constituency_parse(sentence)
            # dependency_doc = self.get_dependency_parse(sentence)
            
            # Find VP-SBAR structures directly
            vp_sbar_structures = self.find_vp_sbar_structures(constituency_tree)
            
            # Analyze each VP-SBAR structure and create simplified output
            embedded_questions = []
            for sbar_info in vp_sbar_structures:
                gap_type = self.analyze_gap_type_hybrid(sbar_info, dependency_doc, sentence)
                
                embedded_questions.append({
                    'selecting_verb': sbar_info['selecting_verb'],
                    'gap_type': gap_type
                })
            
            return embedded_questions
            
        except Exception as e:
            # Return empty list on error
            return []
     