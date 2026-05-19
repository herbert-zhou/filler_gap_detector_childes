from typing import Optional, List
from nltk.tree import Tree


class MatrixQuestionDetector:
    """
    Hybrid detector for matrix questions. Uses both constituency and dependency
    heuristics in parallel and reports agreement or disagreement.
    """

    def __init__(self):
        # Aux inventory for PMQ detection
        self.auxiliaries = {
            'do', 'does', 'did', 'be', 'is', 'am', 'are', 'was', 'were',
            'have', 'has', 'had', 'will', 'would', 'shall', 'should',
            'can', 'could', 'may', 'might', 'must', 'ought', 'need', 'dare'
        }
        # Wh tokens we care about
        self.wh_words = {"who", "what", "where", "when", "why", "how", "which", "whom", "whose"}

    # -------------------------
    # Constituency helpers
    # -------------------------
    def _unwrap_top(self, tree: Tree) -> Tree:
        """If the tree has a top-level 'TOP' or 'ROOT' node with a single child, unwrap it and return that child. Otherwise, return the tree as is."""
        if tree.label() in {"TOP", "ROOT"} and len(tree) == 1 and isinstance(tree[0], Tree):
            return tree[0]
        return tree

    def _find_interrogative_root(self, tree: Tree) -> Optional[Tree]:
        """Find the root of the interrogative clause in the tree. Only consider the given root and its child (no recursion) -- assuming that for MatrixQ, the interrogative clause is at the root or one level below the root."""
        if tree.label() in {'SBARQ', 'SQ', 'SINV'}:
            return tree
        for child in tree:
            if isinstance(child, Tree) and child.label() in {'SBARQ', 'SQ', 'SINV'}:
                return child
        return None
    
    # def _find_all_interrogative_roots(self, tree: Tree) -> list:
    #     """
    #     Recursively find all subtrees rooted in SBARQ, SQ, or SINV.
    #     Returns a list ordered top-to-bottom (preorder traversal).
    #     """
    #     results = []
    #     if tree.label() in {"SBARQ", "SQ", "SINV"}:
    #         results.append(tree)
    #     for child in tree:
    #         if isinstance(child, Tree):
    #             results.extend(self._find_all_interrogative_roots(child))
    #     return results


    def _sbarq_only_wh(self, sbarq: Tree) -> bool:
        """Plain wh-questions like 'Who?'"""
        core_children = [
            c for c in sbarq
            if isinstance(c, Tree) and c.label() not in {'.', ',', ':', 'PRN'}
        ]
        return len(core_children) == 1 and core_children[0].label().startswith('WH')

    def _find_wh_phrase_label(self, tree: Tree) -> Optional[str]:
        if tree.label().startswith('WH'):
            return tree.label()
        for child in tree:
            if isinstance(child, Tree):
                lab = self._find_wh_phrase_label(child)
                if lab:
                    return lab
        return None

    def _first_word(self, tree: Tree) -> str:
        if isinstance(tree, str):
            return tree
        if not tree:
            return ""
        child = tree[0]
        if isinstance(child, str):
            return child
        return self._first_word(child)

    def get_children(self, node: Tree):
        """Return immediate children as (label, subtree_or_token)."""
        children = []
        for child in node:
            if hasattr(child, "label"):
                children.append((child.label(), child))
            else:
                children.append(("TOKEN", child))
        return children
    
    # def analyze_sq_children(self, children):
    #     """
    #     Given ordered children of an SQ/SINV node (list of (label, node)),
    #     return one of:
    #     - 'AUX+NP'     : first MD/V* followed (to the right) by an NP before any VP
    #     - 'AUX+X_noNP' : first MD/V* is not followed by NP, or a VP intervenes before NP
    #     - None         : no clear verbal/modal anchor

    #     Notes:
    #     - Labels checked: 'MD' and any label starting with 'V' (incl. 'VP', 'VBD', 'VBZ', ...)
    #     - We only look at siblings to the *right* of the first MD/V*.
    #     """

    #     # 1) Find the first MD or V* (incl. VP)
    #     first_idx = None
    #     for i, (lbl, _) in enumerate(children):
    #         if lbl == "MD" or lbl.startswith("V"):
    #             first_idx = i
    #             break

    #     if first_idx is None:
    #         # No clear verbal/modal anchor; fall back: VP-without-NP at this layer → SMQ shape
    #         has_vp = any(lbl == "VP" for lbl, _ in children)
    #         has_np = any(lbl == "NP" for lbl, _ in children)
    #         if has_vp and not has_np:
    #             return "AUX+X_noNP"
    #         return None

    #     # 2) Inspect only the siblings to the right of the first MD/V*
    #     right = [lbl for lbl, _ in children[first_idx + 1:]]

    #     # If there's no right sibling at all (e.g., only child is VP), treat as no NP present
    #     if not right:
    #         return "AUX+X_noNP"

    #     # Find the first occurrences (if any)
    #     try:
    #         idx_np = right.index("NP")
    #     except ValueError:
    #         idx_np = None

    #     try:
    #         idx_vp = right.index("VP")
    #     except ValueError:
    #         idx_vp = None

    #     # Decide by relative order
    #     if idx_np is not None:
    #         if idx_vp is not None and idx_vp < idx_np:
    #             # A VP intervenes before the first NP → no explicit subject at this layer
    #             return "AUX+X_noNP"
    #         # First NP appears before any VP → explicit subject present
    #         return "AUX+NP"
    #     else:
    #         # No NP at all to the right
    #         return "AUX+X_noNP"

    def analyze_sq_children(self, children):
        """
        Given ordered children of an SQ/SINV node (list of (label, node)),
        return a dict:
        {
            'S':  'NPVP' | 'VP' | None,   # surface pattern under SQ
            'CC': 'NPVP' | 'VP' | None    # immediate VP-internal SBAR→S shape (embedded clause)
        }

        Semantics:
        S: 'NPVP'   ≈ legacy 'AUX+NP'
        S: 'VP'     ≈ legacy 'AUX+X_noNP'
        S: None     ≈ legacy None

        CC: 'VP'      => SBAR→S lacks NP  (embedded S -> VP)   → SUBJECT gap
        CC: 'NPVP'    => SBAR→S has NP+VP (embedded S -> NP VP)→ NON-SUBJ gap (obj/obl)
        CC: None      => no immediate VP→SBAR→S pattern detected
        """

        # ---- helpers over benepar Tree nodes ----
        def _is_tree(node):
            return hasattr(node, "label")

        def _first_child_with_label(node, want_lbl):
            if not _is_tree(node):
                return None
            for ch in node:
                if _is_tree(ch) and ch.label() == want_lbl:
                    return ch
            return None

        def _has_immediate_child_label(node, want_lbl):
            if not _is_tree(node):
                return False
            return any(_is_tree(ch) and ch.label() == want_lbl for ch in node)

        # Defaults
        S_label = None
        CC_label = None

        # 1) Find the first MD or V* (incl. VP)
        first_idx = None
        for i, (lbl, _) in enumerate(children):
            if lbl == "MD" or lbl.startswith("V"):
                first_idx = i
                break

        if first_idx is None:
            # No clear verbal/modal anchor; fall back: VP-without-NP at this layer → SMQ shape
            has_vp = any(lbl == "VP" for lbl, _ in children)
            has_np = any(lbl == "NP" for lbl, _ in children)
            if has_vp and not has_np:
                S_label = "VP"   # legacy 'AUX+X_noNP'
            else:
                S_label = None
            return {"S": S_label, "CC": CC_label}

        # 2) Inspect only the siblings to the right of the first MD/V*
        right_slice  = children[first_idx + 1:]
        right_labels = [lbl for lbl, _ in right_slice]
        right_nodes  = [node for _, node in right_slice]

        if not right_labels:
            S_label = "VP"  # legacy 'AUX+X_noNP'
            return {"S": S_label, "CC": CC_label}

        # Find the first occurrences (if any) of NP and VP to the right
        try:
            idx_np = right_labels.index("NP")
        except ValueError:
            idx_np = None

        try:
            idx_vp = right_labels.index("VP")
        except ValueError:
            idx_vp = None

        # Base S label (legacy surface pattern)
        if idx_np is not None:
            if idx_vp is not None and idx_vp < idx_np:
                S_label = "VP"   # VP intervenes before NP → no explicit subject at this layer
            else:
                S_label = "NPVP" # NP appears before VP → explicit subject present
        else:
            S_label = "VP"       # no NP to the right

        # CC detection only runs if we have a VP to the right (we look inside that VP)
        if idx_vp is not None:
            vp_node = right_nodes[idx_vp]
            sbar = _first_child_with_label(vp_node, "SBAR")
            if sbar is not None:
                # Per spec: look for immediate S under SBAR (do not traverse CP here)
                s_under = _first_child_with_label(sbar, "S")
                if s_under is not None:
                    has_np_in_s = _has_immediate_child_label(s_under, "NP")
                    has_vp_in_s = _has_immediate_child_label(s_under, "VP")
                    if has_vp_in_s and not has_np_in_s:
                        CC_label = "VP"       # subject gap (S -> VP)
                    elif has_vp_in_s and has_np_in_s:
                        CC_label = "NPVP"     # non-subject gap (S -> NP VP)
                    else:
                        CC_label = None
                # else: no immediate S → leave CC as None

        return {"S": S_label, "CC": CC_label}

    
    def classify_interrogative(self, qroot: Tree) -> Optional[str]:
        """
        Constituency-based classification.
        qroot is SBARQ / SQ / SINV

        Uses analyze_sq_children(...) which returns:
            {'S': 'NPVP' | 'VP' | None, 'CC': 'NPVP' | 'VP' | None}

        Mapping:
        S:
            'NPVP' -> legacy 'AUX+NP'
            'VP'   -> legacy 'AUX+X_noNP'
            None   -> legacy None

        CC:
            'VP'   -> embedded S -> VP      => subject gap        (CC_SMQ if WHNP; CC_AMQ if WH-adv/PP/adj)
            'NPVP' -> embedded S -> NP VP   => non-subject gap    (CC_OMQ if WHNP; CC_AMQ if WH-adv/PP/adj)
            None   -> no cross-clausal pattern -> fall back to legacy mapping
        """

        # ---------------- Case 1: root is SBARQ ----------------
        if qroot.label() == "SBARQ":
            # e.g., 'What?' / 'Who?' (bare wh)
            if self._sbarq_only_wh(qroot):
                return "PlainMQ"

            # Identify WH phrase category at SBARQ frontier
            wh_label = self._find_wh_phrase_label(qroot)  # e.g., 'WHNP', 'WHADVP', 'WHPP', 'WHADJP', or None

            # Find the embedded SQ/SINV/S
            sq_child = None
            for c in qroot:
                if isinstance(c, Tree) and c.label() in {"SQ", "SINV", "S"}:
                    sq_child = c
                    break
            if sq_child is None:
                return None

            children = self.get_children(sq_child)          # -> List[(label, node)]
            pattern = self.analyze_sq_children(children)     # -> {'S': ..., 'CC': ...}

            S  = pattern.get("S")
            CC = pattern.get("CC")

            # -------- Cross-clausal branch (generalized across WH types) --------
            if CC in {"VP", "NPVP"}:
                # WH adjuncts (ADVP/PP/ADJP) → cross-clausal Adjunct MQ
                if wh_label in {"WHADVP", "WHPP", "WHADJP"}:
                    return "CC_AMQ"

                # Default nominal WH (WHNP and variants) → decide Subj vs Obj
                # (If wh_label is None, treat it as nominal by default)
                if wh_label in {"WHNP", None}:
                    if CC == "VP":
                        return "CC_SMQ"   # subject gap
                    else:  # 'NPVP'
                        return "CC_OMQ"   # object/oblique gap

            # -------- No cross-clausal signal → legacy mapping by S-pattern --------
            if wh_label in {"WHADVP", "WHADJP", "WHPP"}:
                if S == "NPVP":
                    return "AMQ"          # adjunct matrix question with explicit subject
                # If S == "VP", many telegraphic forms won’t fit neatly; keep None

            if wh_label == "WHNP":
                if S == "NPVP":
                    return "OMQ"          # object matrix question (explicit subject present)
                elif S == "VP":
                    return "SMQ"          # subject matrix question (no NP before VP)

            return None

        # ---------------- Case 2: root is SQ or SINV ----------------
        elif qroot.label() in {"SQ", "SINV"}:
            children = self.get_children(qroot)
            pattern = self.analyze_sq_children(children)   # -> {'S': ..., 'CC': ...}
            S = pattern.get("S")

            # If there is no WH among immediate children, treat as polar/inverted MQ
            has_wh_child = any(lbl.startswith("WH") for lbl, _ in children)
            if not has_wh_child:
                if S == "NPVP":
                    return "PMQ"          # polar matrix question (aux + subject)
                elif S == "VP":
                    return "SMQ"          # rare fallback for telegraphic forms

            return None

        return None


    # -------------------------
    # Dependency helpers
    # -------------------------
    def _get_wh_tokens(self, dep) -> List:
        return [tok for tok in dep if tok.text.lower() in self.wh_words]

    def _other_subjects(self, dep, wh_tokens) -> List:
        return [tok for tok in dep if tok.dep_ in {"nsubj", "nsubjpass"} and tok not in wh_tokens]

    def _is_auxiliary_word(self, word: str) -> bool:
        return word.lower() in self.auxiliaries

    def _is_wh_adjunct_phrase(self, wh_label: str) -> bool:
        return wh_label.startswith(("WHADVP", "WHPP", "WHADJP"))
    

    def dep_classify(self, qroot, tree, dep) -> Optional[str]:
        """Dependency-based classification (original logic)."""
        # ---------- Existential "there"-be exception (apply BEFORE other rules) ----------
        # We: (1) find the WH anchor (handles "how many/much NOUN"),
        #     (2) climb to the governing verb (or use qroot),
        #     (3) if that verb has expl=there -> classify as OMQ.

        def _wh_tokens(span):
            for tok in span:
                if tok.tag_.startswith("W"):  # WP, WP$, WRB, WDT, etc.
                    yield tok
                elif tok.lower_ == "how":
                    yield tok  # we'll handle how-many/much below
                    
        def _get_dep_root(dep):
            # Works for spaCy Doc or Span
            try:
                return dep.root
            except AttributeError:
                # very defensive fallback
                for t in dep:
                    if t.dep_ == "ROOT":
                        return t
                return None

        def _np_head_for_how_many(tok):
            # If tok is 'how' and forms 'how many/much NOUN', return that NOUN head.
            if tok.lower_ == "how":
                h = tok.head
                if h.lower_ in ("many", "much"):
                    # common UD: det(NOUN, many) and advmod(many, how)
                    # try head first
                    if h.head.pos_ in ("NOUN", "PROPN", "PRON"):
                        return h.head
                    # else any NOUN child of 'many/much'
                    for ch in h.children:
                        if ch.pos_ in ("NOUN", "PROPN", "PRON"):
                            return ch
            # starting from 'many/much' itself
            if tok.lower_ in ("many", "much"):
                if tok.head.pos_ in ("NOUN", "PROPN", "PRON"):
                    return tok.head
                for ch in tok.children:
                    if ch.pos_ in ("NOUN", "PROPN", "PRON"):
                        return ch
            return None

        def _climb_to_clause_verb(start):
            t = start
            seen = set()
            while t is not None and t.head is not t and t.i not in seen:
                seen.add(t.i)
                if t.pos_ in ("VERB", "AUX"):
                    return t
                t = t.head
                if t.pos_ in ("VERB", "AUX"):
                    return t
                if t.dep_ == "ROOT":
                    return t if t.pos_ in ("VERB", "AUX") else None
            return None
        
        
        if qroot.label() == "SBARQ":
            if self._sbarq_only_wh(qroot):
                return "PlainMQ"

            wh_label = self._find_wh_phrase_label(qroot)
            wh_tokens = self._get_wh_tokens(dep)
            
            # ---------- start existential exception ----------
            # try each WH cue; return OMQ immediately if expl(there) is found
            for wh in _wh_tokens(dep):
                anchor = _np_head_for_how_many(wh) or wh
                # verb = _climb_to_clause_verb(anchor) or (qroot if qroot is not None and qroot.pos_ in ("VERB", "AUX") else None)
                verb = _climb_to_clause_verb(anchor) or _get_dep_root(dep)
                if verb is not None:
                    # existential 'there' typically shows up as expl on the be-verb
                    if any(ch.dep_ == "expl" and ch.lower_ == "there" for ch in verb.children):
                        return "OMQ"
            # ---------- end existential exception ----------

            if wh_tokens:
                wh = sorted(wh_tokens, key=lambda t: t.i)[0]
                wh_dep = wh.dep_
                other_subjects = self._other_subjects(dep, wh_tokens)

                if not other_subjects:
                    return "SMQ"
                if wh_dep in {"nsubj", "nsubjpass"}:
                    return "SMQ"
                elif wh_dep in {"dobj", "obj"}:
                    return "OMQ"
                elif wh_dep in {"advmod", "obl"}:
                    return "AMQ"
                elif wh_dep in {"attr", "oprd"}:
                    return "OMQ" if other_subjects else "SMQ"
                else:
                    if wh_label and self._is_wh_adjunct_phrase(wh_label):
                        return "AMQ"
                    return "OMQ"
            return "OMQ"

        if qroot.label() in {"SQ", "SINV"}:
            first_word = self._first_word(qroot)
            if self._is_auxiliary_word(first_word):
                return "PMQ"
            if len(qroot) > 0 and isinstance(qroot[0], Tree) and qroot[0].label().startswith("V"):
                return "PMQ"
            return "PMQ"
        return None

    
    def detect_matrix_questions(self, sentence: str, tree: Tree, dep) -> Optional[str]:
        """
        Hybrid detection:
        - Collect all SBARQ/SQ/SINV nodes (qroots), top to bottom.
        - For each qroot, get constituency and dependency decisions.
        - Resolution policy:
            * If exactly one is non-None → return it.
            * If both non-None and agree → return it.
            * If both non-None and disagree → return constituency decision.
            * If both None → continue to next qroot.
        - If no qroot yields a decision → return None.
        """

        tree = self._unwrap_top(tree)
        qroot = self._find_interrogative_root(tree)
        if not qroot:
            return None

        # for qroot in qroots:
        cons_decision = self.classify_interrogative(qroot)
        dep_decision = self.dep_classify(qroot, tree, dep)

        # Case 1: one is None, the other not
        if cons_decision is None and dep_decision is not None:
            print(f"ONLY Dep: {dep_decision} | sentence: {sentence}")
        if dep_decision is None and cons_decision is not None:
            print(f"ONLY Cons: {cons_decision} | sentence: {sentence}")

        # Case 2: both non-None
        if cons_decision is not None and dep_decision is not None:
            if cons_decision == dep_decision:
                return ;
            else: # both non-None but disagree
                print(f"Cons={cons_decision}|Dep={dep_decision}; sentence: {sentence}")
                # if cons_decision == 'SMQ': return dep_decision
                # else: return cons_decision

        # Case 3: both None → move on
        if cons_decision is None and dep_decision is None:
            return ;

