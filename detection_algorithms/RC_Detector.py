from typing import List, Tuple, Dict, Optional, Set
from nltk.tree import Tree

NOUNLIKE_POS = {"NOUN", "PROPN", "PRON"}  # spaCy pos_

class RelativeClauseDetector:
    """
    Dependency-driven Relative Clause detector with a clean 1->2->3->4 workflow.

    Now includes a dependency double-check for SRC (wh-subject relatives):
      - WH token must be nsubj/nsubjpass
      - Its head must be the RC verb with dep 'relcl' and lie under the SBAR's VP
      - That RC verb's head must be a nounlike token in the NP modified by the SBAR
    """

    # --------------------
    # Public entry point
    # --------------------
    def detect_relative_clauses(self, sentence: str, tree, doc=None) -> List[Dict]:
        """
        1) NP->NP SBAR: if WH present -> classify (with SRC dep-check if doc is provided);
           if no WH and S has NP before VP -> queue for reduced ORC/ARC checks.
        2) Detect reduced SRC via NP->NP (VP|ADJP) once, with dep checks.
        3) For queued SBARs: exclude IN/TO before S; ARC_reduced (tight rule); otherwise ORC_reduced.
        4) Normalize & dedup.
        """
        results: List[Dict] = []
        orc_like_no_wh: List[Tuple[Tree, Tree]] = []

        # ---- Step 1: Standard RCs (NP -> NP SBAR) ----
        np_sbar_pairs = self.find_np_sbar_patterns(tree)
        for np_node, sbar in np_sbar_pairs:
            wh_word = self.get_wh_word(sbar)
            svp = self._extract_s_and_vp_under_sbar(sbar)

            if wh_word:
                gap_type = self.classify_relative_clause(np_node, sbar, doc=doc)
                # Skip SRC if its dependency check failed
                if gap_type != 'SRC_dep_fail':
                    results.append({'wh_word': wh_word, 'gap_type': gap_type})
            else:
                if svp is not None:
                    s_node, vp_node = svp
                    np_pos = vp_pos = None
                    for i, ch in enumerate(s_node):
                        if isinstance(ch, Tree) and ch.label() == 'NP' and np_pos is None:
                            np_pos = i
                        if isinstance(ch, Tree) and ch.label() == 'VP' and vp_pos is None:
                            vp_pos = i
                    if np_pos is not None and vp_pos is not None and np_pos < vp_pos:
                        orc_like_no_wh.append((np_node, sbar))

        # ---- Step 2: Reduced SRC in one pass (requires doc) ----
        if doc is not None:
            results.extend(self.detect_reduced_SRC(tree, doc))

        # ---- Step 3: Reduced ORC/ARC from queued SBARs (requires doc) ----
        if doc is not None:
            for np_node, sbar in orc_like_no_wh:
                # 3.1 Exclusion for IN/TO before S
                if self._sbar_has_leading_marker_excluding_wh(sbar):
                    continue

                svp = self._extract_s_and_vp_under_sbar(sbar)
                if svp is None:
                    continue
                s_node, vp_node = svp

                # map leaves
                np_idx = self._leaf_token_indices(doc, np_node)
                vp_idx = self._leaf_token_indices(doc, vp_node)
                if not np_idx or not vp_idx:
                    continue

                # Identify RC verb(s): tokens in VP with dep relcl/acl:relcl whose head is a nounlike token in NP
                rc_verbs = []
                for i in vp_idx:
                    t = doc[i]
                    if t.dep_ in {"relcl","acl:relcl"}:
                        head = t.head
                        if head.i in np_idx and head.pos_ in NOUNLIKE_POS:
                            rc_verbs.append(i)

                # Tight ARC_reduced: object must be DIRECT CHILD of RC verb, incl prep->pobj linked to RC verb
                is_arc_reduced = False
                if rc_verbs:
                    rc_set = set(rc_verbs)
                    for i in vp_idx:
                        t = doc[i]
                        # direct object
                        if t.dep_ in {"dobj","obj"} and t.head.i in rc_set:
                            is_arc_reduced = True
                            break
                        # prepositional object: pobj whose preposition is a child of RC verb
                        if t.dep_ == "pobj":
                            prep = t.head
                            if prep.dep_ == "prep" and prep.head.i in rc_set:
                                is_arc_reduced = True
                                break

                if is_arc_reduced:
                    results.append({'wh_word': None, 'gap_type': 'ARC_reduced'})
                    continue

                # ORC_reduced:
                found_orc = False
                if rc_verbs:
                    found_orc = True
                else:
                    # fallback: any VP token with dep ROOT or parataxis
                    if any(doc[i].dep_ in {"ROOT","parataxis"} for i in vp_idx):
                        found_orc = True

                if found_orc:
                    results.append({'wh_word': None, 'gap_type': 'ORC_reduced'})

        # ---- Step 4: Normalize & dedup ----
        for r in results:
            if r.get('gap_type') == 'RRC_SRC':
                r['gap_type'] = 'SRC_reduced'

        dedup: List[Dict] = []
        seen = set()
        for r in results:
            key = (r.get('wh_word'), r.get('gap_type'))
            if key not in seen:
                seen.add(key)
                dedup.append(r)
        return dedup

    # --------------------------------------------
    # Step 1: NP -> NP SBAR finder & classifiers
    # --------------------------------------------
    def find_np_sbar_patterns(self, tree: Tree) -> List[Tuple[Tree, Tree]]:
        patterns: List[Tuple[Tree, Tree]] = []

        def search(node):
            if isinstance(node, Tree) and node.label() in ['NP', 'FRAG'] and len(node) >= 2:
                # find first NP child and following SBAR sibling
                first_np_idx = None
                for i, child in enumerate(node):
                    if isinstance(child, Tree) and child.label() == 'NP':
                        first_np_idx = i
                        break
                if first_np_idx is not None:
                    for j in range(first_np_idx + 1, len(node)):
                        child = node[j]
                        if isinstance(child, Tree) and child.label() == 'SBAR':
                            patterns.append((node[first_np_idx], child))
                            break
            if isinstance(node, Tree):
                for ch in node:
                    search(ch)

        search(tree)
        return patterns

    def get_wh_word(self, sbar: Tree) -> Optional[str]:
        wh_node = self._find_preS_wh_constituent(sbar)
        if wh_node is None:
            return None
        head = self._extract_leftmost_wh_head(wh_node)
        if head is None:
            phrase = self._leaves_text(wh_node)
            if phrase:
                head = phrase.split()[0]
        return head

    def classify_relative_clause(self, np_node: Tree, sbar: Tree, doc=None) -> str:
        """
        Classify WH-present RC as PRC/ARC/SRC/ORC, with a dependency check for SRC when doc is provided.
        """
        # WH type
        wh_node = self._find_preS_wh_constituent(sbar)
        wh_label = wh_node.label() if isinstance(wh_node, Tree) else None
        wh_head = self._extract_leftmost_wh_head(wh_node) if wh_node is not None else None

        # Possessive WH (whose / WP$)
        if wh_label == 'WHNP':
            leaves = wh_node.leaves()
            if any(x.lower() == 'whose' for x in leaves):
                return 'PRC'
        if wh_label == 'WP$' or wh_head == 'whose':
            return 'PRC'

        # Adjunct-like: WHADVP, WHPP, or classic adverbial wh heads
        if wh_label in {'WHADVP', 'WHPP'}:
            return 'ARC'
        if wh_head in {'where', 'when', 'why', 'how'}:
            return 'ARC'

        # SRC vs ORC by S structure
        svp = self._extract_s_and_vp_under_sbar(sbar)
        if svp is None:
            # default
            guess = 'SRC'
        else:
            s_node, _ = svp
            np_pos = vp_pos = None
            for i, ch in enumerate(s_node):
                if isinstance(ch, Tree) and ch.label() == 'NP' and np_pos is None:
                    np_pos = i
                if isinstance(ch, Tree) and ch.label() == 'VP' and vp_pos is None:
                    vp_pos = i
            guess = 'ORC' if (np_pos is not None and vp_pos is not None and np_pos < vp_pos) else 'SRC'

        # If guess is SRC and doc is provided, run dependency validation
        if guess == 'SRC' and doc is not None:
            if not self._validate_src_dependency(np_node, sbar, doc):
                return 'SRC_dep_fail'
        # If guess is ORC and doc is provided, run dependency validation
        if guess == 'ORC' and doc is not None:
            if not self._validate_orc_dependency(np_node, sbar, doc):
                return 'ORC_dep_fail'
        return guess

    # --------------------------------------------
    # Step 2: Reduced SRC (one-pass finder; dep-only checks)
    # --------------------------------------------
    def find_np_mod_patterns(self, tree: Tree) -> List[Tuple[Tree, str, Tree]]:
        patterns: List[Tuple[Tree, str, Tree]] = []

        def search(node):
            if isinstance(node, Tree):
                if node.label() == 'NP' and len(node) >= 2:
                    # first NP child and first following VP/ADJP sibling
                    first_np_idx = None
                    for i, child in enumerate(node):
                        if isinstance(child, Tree) and child.label() == 'NP':
                            first_np_idx = i
                            break
                    if first_np_idx is not None:
                        for j in range(first_np_idx + 1, len(node)):
                            child = node[j]
                            if isinstance(child, Tree) and child.label() in {'VP', 'ADJP'}:
                                patterns.append((node[first_np_idx], child.label(), child))
                                break
                for ch in node:
                    search(ch)

        search(tree)
        return patterns

    def detect_reduced_SRC(self, tree: Tree, doc) -> List[Dict]:
        out: List[Dict] = []
        patterns = self.find_np_mod_patterns(tree)
        for np_node, label, mod_node in patterns:
            np_idx = self._leaf_token_indices(doc, np_node)
            mod_idx = self._leaf_token_indices(doc, mod_node)
            if not np_idx or not mod_idx:
                continue
            if label == 'VP':
                if any(doc[i].dep_ == "acl" and doc[doc[i].head.i].pos_ in NOUNLIKE_POS and doc[i].head.i in np_idx
                       for i in mod_idx):
                    out.append({'wh_word': None, 'gap_type': 'SRC_reduced'})
            elif label == 'ADJP':
                if any(doc[i].dep_ == "amod" and doc[doc[i].head.i].pos_ in NOUNLIKE_POS and doc[i].head.i in np_idx
                       for i in mod_idx):
                    out.append({'wh_word': None, 'gap_type': 'SRC_reduced'})
        return out
    
    def _leaf_token_indices(self, doc, node: Tree) -> List[int]:
        """
        Map the leaves dominated by `node` to spaCy token indices.

        This implementation aligns based on the *character stream* formed
        by the subtree leaves vs. the full doc tokens, which makes it robust
        to tokenization differences like 'wan' + 'na' vs. 'wanna'.

        Steps:
          1) Concatenate node.leaves() into a string with no spaces.
          2) Concatenate doc tokens (t.text) into a string with no spaces,
             while tracking start-char offsets for each token.
          3) Find the subtree string as a substring of the doc string.
          4) Return all token indices whose char-span overlaps that substring.

        Caveat:
          - If the same character sequence appears multiple times in the
            sentence, we take the *first* match (ambiguity is unavoidable
            in that case without extra context).
        """
        if doc is None or not isinstance(node, Tree):
            return []

        # 1) Subtree leaves -> continuous string (no spaces)
        leaves = [str(x) for x in node.leaves()]
        if not leaves:
            return []
        sub_str = "".join(leaves)

        # 2) Doc tokens -> continuous string (no spaces) + token start offsets
        doc_texts = [t.text for t in doc]
        doc_stream = []
        token_start = []  # start char index of each token in doc_stream
        cur = 0
        for tok_text in doc_texts:
            token_start.append(cur)
            doc_stream.append(tok_text)
            cur += len(tok_text)
        doc_str = "".join(doc_stream)

        # 3) Find subtree string inside the doc string
        start_char = doc_str.find(sub_str)
        if start_char == -1:
            # Could not align this subtree reliably
            return []

        end_char = start_char + len(sub_str)  # end is exclusive

        # 4) Collect all token indices whose [start, end) overlaps [start_char, end_char)
        idxs: List[int] = []
        for i, tok_text in enumerate(doc_texts):
            ts = token_start[i]
            te = ts + len(tok_text)  # exclusive
            # overlap if token span and subtree span intersect
            if te <= start_char:
                continue
            if ts >= end_char:
                break
            idxs.append(i)

        return idxs




    def _find_preS_wh_constituent(self, sbar: Tree) -> Optional[Tree]:
        if not isinstance(sbar, Tree):
            return None
        for child in sbar:
            if isinstance(child, Tree) and child.label() == 'S':
                break
            if not isinstance(child, Tree):
                continue
            lab = child.label()
            if lab in {',', ':', ';', '-LRB-', '-RRB-', '``', "''"}:
                continue
            if lab.startswith('WH'):
                return child
            if lab in {'WDT', 'WP', 'WP$', 'WRB'}:
                return child
            if lab in {'IN', 'WDT'} and child.height() == 2:
                tok = child[0]
                if isinstance(tok, str) and tok.lower() == 'that':
                    return child
        return None

    def _extract_leftmost_wh_head(self, node: Optional[Tree]) -> Optional[str]:
        if node is None:
            return None
        if node.height() == 2 and node.label() in {'WDT', 'WP', 'WP$', 'WRB'}:
            return node[0]
        for child in node:
            if isinstance(child, Tree):
                head = self._extract_leftmost_wh_head(child)
                if head:
                    return head
        if node.height() == 2 and node.label() in {'IN', 'WDT'}:
            tok = node[0]
            if isinstance(tok, str) and tok.lower() == 'that':
                return 'that'
        return None

    def _leaves_text(self, node: Tree) -> str:
        if not isinstance(node, Tree):
            return ""
        leaves = node.leaves()
        txt = " ".join(leaves)
        fixes = {" ,": ",", " .": ".", " !": "!", " ?": "?", " :": ":", " ;": ";",
                 " )": ")", "( ": "(", " n't": "n't", " ’": "’", " '": "'",
                 "-LRB- ": "(", " -RRB-": ")"}
        for k, v in fixes.items():
            txt = txt.replace(k, v)
        return txt.strip()

    def _extract_s_and_vp_under_sbar(self, sbar: Tree) -> Optional[Tuple[Tree, Tree]]:
        if not isinstance(sbar, Tree):
            return None
        s_node = None
        for ch in sbar:
            if isinstance(ch, Tree) and ch.label() == 'S':
                s_node = ch
                break
        if s_node is None:
            return None
        vp_node = None
        for ch in s_node:
            if isinstance(ch, Tree) and ch.label() == 'VP':
                vp_node = ch
                break
        if vp_node is None:
            return None
        return (s_node, vp_node)

    def _sbar_has_leading_marker_excluding_wh(self, sbar: Tree) -> bool:
        if not isinstance(sbar, Tree):
            return False
        first_s = None
        for i, ch in enumerate(sbar):
            if isinstance(ch, Tree) and ch.label() == 'S':
                first_s = i
                break
        if first_s is None:
            return False
        for i in range(first_s):
            ch = sbar[i]
            if isinstance(ch, Tree) and ch.label() in {'IN', 'TO'}:
                return True
        return False

    # ----------- SRC dependency validation -----------
    def _validate_src_dependency(self, np_node: Tree, sbar: Tree, doc) -> bool:
        """
        Require:
          1) WH token is nsubj/nsubjpass
          2) Its head is an RC verb (dep 'relcl') and lies under VP of this SBAR
          3) That RC verb's head is a nounlike token in the NP being modified
        """
        if doc is None:
            return True  # can't validate, so don't block
        
        wh_node = self._find_preS_wh_constituent(sbar)
        if wh_node is None:
            return False

        # map WH phrase to tokens and pick the WH token serving as subject
        wh_idx = self._leaf_token_indices(doc, wh_node)
        wh_idx = [idx for idx in wh_idx if doc[idx].tag_.startswith('W')]  # filter to WH POS tags
        
        wh_subj_idx = None
        for i in wh_idx:
            if doc[i].dep_ in {"nsubj","nsubjpass"}:
                wh_subj_idx = i
                break
        if wh_subj_idx is None:
            return False

        # RC verb should be the head of the WH subject and have dep relcl
        rc_verb = doc[wh_subj_idx].head
        if rc_verb.dep_ != "relcl":
            return False

        # RC verb must be inside the VP under this SBAR
        svp = self._extract_s_and_vp_under_sbar(sbar)
        if svp is None:
            return False
        _, vp_node = svp
        vp_idx = self._leaf_token_indices(doc, vp_node)
        if rc_verb.i not in vp_idx:
            return False

        # RC verb's head must be a nounlike token in the modified NP
        np_idx = self._leaf_token_indices(doc, np_node)
        if rc_verb.head.i not in np_idx:
            return False

        return True

    def _validate_orc_dependency(self, np_node: Tree, sbar: Tree, doc) -> bool:
        '''
        Dependency validator for Object Relative Clauses (ORC) when a WH-word is present.

        Conditions:
          1) WH token inside SBAR has dep_ in {'dobj','obj','pobj','dative'}.
          2) There exists an RC verb with dep_ in {'relcl','acl:relcl'} under this SBAR.
          3) Let X be that RC verb's head. Either:
               (a) X itself is a nounlike token in the modified NP, or
               (b) X dominates (via child dependencies) a nounlike token in that NP.
          4) The WH token depends (along its head chain) on that RC verb.
        '''
        if doc is None:
            return True  # can't validate, so don't block

        # Collect indices for NP and SBAR subtrees
        np_idx = set(self._leaf_token_indices(doc, np_node))
        sbar_idx = set(self._leaf_token_indices(doc, sbar))

        # Find a WH token inside SBAR
        wh_candidates = [doc[i] for i in sorted(sbar_idx) if doc[i].tag_.startswith('W')]
        if not wh_candidates:
            return False
        wh_tok = wh_candidates[0]

        # 1) WH token must be object-like
        if wh_tok.dep_ not in {'dobj','pobj','dative'}:
            return False

        # 2) Identify RC verb candidates (relcl/acl:relcl) inside this SBAR
        rc_verbs = []
        for i in sorted(sbar_idx):
            t = doc[i]
            if t.dep_ in {'relcl', 'acl:relcl'}:
                rc_verbs.append(t)
        if not rc_verbs:
            return False

        # 2b) If an S->VP exists under SBAR, prefer rc_verbs that lie under that VP
        svp = self._extract_s_and_vp_under_sbar(sbar)
        if svp is not None:
            _, vp_node = svp
            vp_idx = set(self._leaf_token_indices(doc, vp_node))
            rc_verbs_vp = [v for v in rc_verbs if v.i in vp_idx]
            if rc_verbs_vp:
                rc_verbs = rc_verbs_vp

        # Take the first candidate RC verb
        rc_verb = rc_verbs[0]

        # 3) Check the head X of the RC verb against the modified NP, with relaxed dominance
        X = rc_verb.head

        # Build children map for downward traversal from X
        children = {}
        for tok in doc:
            children.setdefault(tok.head.i, []).append(tok)

        def dominates_np_nounlike(root_tok):
            '''Return True if root_tok or any of its descendants is a nounlike token in the NP.'''
            stack = [root_tok]
            seen = set()
            while stack:
                cur = stack.pop()
                if cur.i in seen:
                    continue
                seen.add(cur.i)
                if cur.i in np_idx and cur.pos_ in NOUNLIKE_POS:
                    return True
                for ch in children.get(cur.i, []):
                    if ch.i not in seen:
                        stack.append(ch)
            return False

        if not dominates_np_nounlike(X):
            return False

        # 4) Ensure WH token depends (directly or along head-chain) on the RC verb
        cur = wh_tok
        seen = set()
        ok_chain = False
        for _ in range(20):  # climb up to root with a safety bound
            if cur.i == rc_verb.i:
                ok_chain = True
                break
            seen.add(cur.i)
            if cur.head.i == cur.i or cur.head.i in seen:
                break
            cur = cur.head

        if not ok_chain:
            return False

        return True


        
