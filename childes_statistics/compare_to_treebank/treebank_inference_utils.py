"""
Extractor + classifier for CHILDES Treebank-style PTB trees with -NONE- traces.

This script reads a `.parsed` file that contains many Penn Treebank (PTB)-style bracketed
trees (often spanning multiple lines). It performs three core tasks:

1) Sentence extraction:
   - Walk preterminals (POS -> word) and drop empty-category tokens whose POS labels
     begin with "-NONE-".
   - Apply minimal detokenization (punctuation and common clitics).

2) Dependency extraction (trace-based):
   - Find trace tokens like "*T*-1" under a "-NONE-..." preterminal.
   - For each trace, identify the *gap phrase* (parent phrase containing the -NONE- token).
   - Link the trace to a *filler phrase* (usually a WH* phrase), using:
       a) coindex matching (e.g., WHNP-1), or
       b) a fallback that picks a WH* phrase inside the nearest dominating SBAR/SBARQ.

3) Classification:
   - construction ∈ {MATRIX_Q, EMBEDDED_Q, REL_CLAUSE, OTHER}
   - gap_subtype ∈ {SUBJECT, OBJECT, ADJUNCT, POBJ, PP, NP_OTHER, ...}
   - final_label = f"{construction}_{gap_subtype}"

Outputs:
  - <prefix>_sentences.csv: one row per tree: tree_id, sentence, has_trace, n_deps
  - <prefix>_dependencies.csv: one row per dependency (trace): linked filler/gap + labels
  - <prefix>_tree_final_classification.csv: one row per tree: sentence + aggregated labels

Note:
  The classification heuristics are intentionally simple and structural; they are meant
  to be tuned to match your research definitions and/or the treebank's annotation scheme.
"""

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any, Tuple

import pandas as pd
from nltk.tree import ParentedTree
from collections import Counter

# -----------------------------
# Reading: iterate over trees by paren balancing
# -----------------------------
def iter_tree_strings(path: Path) -> Iterator[str]:
    """
    Yield one PTB-style bracketed tree string at a time by balancing parentheses.

    The CHILDES Treebank `.parsed` files typically contain many trees, each spanning
    multiple lines and without a consistent delimiter. This function segments the file
    robustly by scanning the entire text and tracking parenthesis depth:

      - increment depth on '('
      - decrement depth on ')'
      - when depth returns to 0, a full tree has ended

    Args:
        path: Path to a file containing concatenated bracketed parse trees.

    Yields:
        A string containing a single complete bracketed tree, including the outermost
        parentheses (e.g., "(ROOT ... )").

    Raises:
        OSError: If the file cannot be read.
    """
    text = path.read_text(encoding="utf8", errors="replace")
    buf: List[str] = []
    depth = 0
    started = False

    for ch in text:
        if ch == "(":
            depth += 1
            started = True
        if started:
            buf.append(ch)
        if ch == ")":
            depth -= 1
            if started and depth == 0:
                s = "".join(buf).strip()
                if s:
                    yield s
                buf = []
                started = False


# -----------------------------
# Label helpers
# -----------------------------
ANGLE_TAG_RE = re.compile(r"<[^>]+>")
INDEX_RE = re.compile(r"-(\d+)(?=-|<|$)")         # matches "-1-" or "-1<" or "-1" end
TRACE_WORD_RE = re.compile(r"^\*(?P<tracekind>T|ICH|RNR|U|EXP)\*-(?P<idx>\d+)$")


def strip_angle_tags(label: str) -> str:
    """
    Remove CHILDES Treebank angle-bracket metadata from a label.

    The CHILDES Treebank adds information such as animacy/theta roles in angle brackets,
    e.g. "WHNP-1-<INANIM>-<SUBJMATT-V1>". For most syntactic decisions we want to ignore
    these decorations while keeping core PTB dash tags (e.g., "-SBJ", "-TMP") and indices.

    Args:
        label: A tree node label, possibly containing "<...>" metadata.

    Returns:
        The label with any "<...>" substrings removed.
    """
    return ANGLE_TAG_RE.sub("", label)


def base_cat(label: str) -> str:
    """
    Extract the base syntactic category from a node label.

    Examples:
        "WHNP-1-<INANIM>-<SUBJMATT>" -> "WHNP"
        "NP-SBJ-2"                    -> "NP"
        "VP"                          -> "VP"

    Args:
        label: A tree node label.

    Returns:
        The base category (string before the first '-'), after removing angle tags.
    """
    return strip_angle_tags(label).split("-")[0]


def extract_index(label: str) -> Optional[int]:
    """
    Extract PTB-style coindexation number from a node label, if present.

    PTB often marks coindexation on moved constituents and traces via integers:
        - filler: WHNP-1
        - trace:  *T*-1 (handled elsewhere)

    This function looks for a dash-number sequence in the label:
        "...-1..." -> 1

    Args:
        label: A node label that may contain a coindexation number.

    Returns:
        The integer index if one is found, otherwise None.
    """
    m = INDEX_RE.search(label)
    return int(m.group(1)) if m else None


# -----------------------------
# Sentence extraction
# -----------------------------
def extract_sentence(tree_str: str) -> str:
    t = ParentedTree.fromstring(tree_str)

    toks: List[str] = []
    for preterm in t.subtrees(lambda x: x.height() == 2):
        pos = preterm.label()
        child = preterm[0]

        # Standard empty categories: (-NONE- *T*-1), (-NONE- *PRO*), etc.
        if pos.startswith("-NONE-"):
            continue

        # Some corpora have weird unary wrappers like (NP ... (-NONE-*PRO*))
        # where the "child" is itself a Tree, not a string token.
        if not isinstance(child, str):
            # If it *does* contain exactly one string leaf, keep it; otherwise skip.
            leaves = [w for w in child.leaves() if isinstance(w, str)] if isinstance(child, ParentedTree) else []
            if len(leaves) == 1:
                tok = leaves[0]
            else:
                continue
        else:
            tok = child

        # Extra guard for fused labels like (-NONE-*PRO*) showing up as a token
        if isinstance(tok, str) and tok.startswith("-NONE-"):
            continue

        toks.append(tok)

    # minimal detokenization
    no_space_before = {",", ".", "?", "!", ":", ";", ")", "]", "}", "''"}
    no_space_after = {"(", "[", "{", "``"}

    out: List[str] = []
    for tok in toks:
        if out and tok in no_space_before:
            out[-1] += tok
        elif out and out[-1] in no_space_after:
            out[-1] += tok
        elif out and tok in {"'s", "n't", "'re", "'ve", "'ll", "'d"}:
            out[-1] += tok
        else:
            out.append(tok)

    return " ".join(out)


# -----------------------------
# Tree navigation helpers
# -----------------------------
def nearest_ancestor(node: ParentedTree, pred) -> Optional[ParentedTree]:
    """
    Return the nearest ancestor of `node` satisfying predicate `pred`.

    Args:
        node: A ParentedTree node from which to start walking upward.
        pred: A callable that takes a ParentedTree node and returns bool.

    Returns:
        The closest ancestor node for which pred(ancestor) is True, or None if no such
        ancestor exists.

    Example:
        nearest_ancestor(trace_node, lambda x: base_cat(x.label()) in {"SBAR", "SBARQ"})
    """
    p = node.parent()
    while p is not None:
        if pred(p):
            return p
        p = p.parent()
    return None


def has_ancestor(node: ParentedTree, pred) -> bool:
    """
    Test whether `node` has any ancestor satisfying predicate `pred`.

    Args:
        node: A ParentedTree node.
        pred: Predicate over ParentedTree nodes.

    Returns:
        True iff there exists an ancestor a of node such that pred(a) is True.
    """
    return nearest_ancestor(node, pred) is not None


# -----------------------------
# Filler linking
# -----------------------------
def find_filler_by_index(root: ParentedTree, idx: int) -> Optional[ParentedTree]:
    """
    Find a likely filler constituent by coindexation number.

    Primary linking strategy for PTB traces:
      - If a trace token is "*T*-idx", then some constituent label often contains "-idx",
        e.g. "WHNP-idx", "NP-idx", etc.
      - We collect all subtrees with that label index and choose the best candidate.

    Selection heuristic (when multiple candidates share the index):
      - prefer WH* phrases (WHNP/WHADVP/WHPP/...)
      - prefer larger constituents (approx. by subtree size)

    Args:
        root: Root of the ParentedTree for one sentence.
        idx: Coindexation integer from a trace token.

    Returns:
        A ParentedTree node representing the best filler candidate, or None if no
        indexed constituent is found.

    Notes:
        Some CHILDES Treebank sentences omit the coindexation on the filler. In that
        case, this function returns None and you should use `find_wh_in_scope`.
    """
    cands = [st for st in root.subtrees() if extract_index(st.label()) == idx]
    if not cands:
        return None

    def score(st: ParentedTree) -> Tuple[int, int]:
        wh_bonus = 2 if base_cat(st.label()).startswith("WH") else 0
        size = len(list(st.subtrees()))
        return (wh_bonus, size)

    return max(cands, key=score)


def find_wh_in_scope(trace_preterm: ParentedTree) -> Optional[ParentedTree]:
    """
    Fallback filler linker: choose a WH* phrase inside the local SBAR/SBARQ scope.

    This is used when filler coindexation is missing or unreliable. The heuristic is:

      1) find the nearest dominating SBAR or SBARQ that contains the trace
      2) among WH* phrases inside that scope, choose the "best" one

    Candidate scoring:
      - prefer larger WH phrases (approx. by subtree size)
      - penalize fillers containing "*NULL*" leaves (often used for null relativizers)

    Args:
        trace_preterm: The preterminal node whose label begins with "-NONE-" and whose
            word matches "*T*-k" (or similar).

    Returns:
        A WH* subtree candidate, or None if no such WH* phrase exists in the nearest SBAR/SBARQ.
    """
    scope = nearest_ancestor(trace_preterm, lambda x: base_cat(x.label()) in {"SBARQ", "SBAR"})
    if scope is None:
        return None

    wh_cands = [st for st in scope.subtrees() if base_cat(st.label()).startswith("WH")]
    if not wh_cands:
        return None

    def score(st: ParentedTree) -> Tuple[int, int]:
        leaves = [w for w in st.leaves() if isinstance(w, str)]
        null_penalty = -1 if any(w == "*NULL*" for w in leaves) else 0
        size = len(list(st.subtrees()))
        return (size, null_penalty)

    return max(wh_cands, key=score)


# -----------------------------
# Construction classification
# -----------------------------
def trace_family(none_pos: str) -> str:
    """
    Coarsely classify trace families based on the -NONE- POS label.

    In the CHILDES Treebank sample, common labels include:
      - "-NONE-ABAR-WH-"  (wh-related traces)
      - "-NONE-ABAR-RC-"  (relative-clause traces)
      - "-NONE-A-PASS-"   (passive)
      - "-NONE-A-RAISE-"  (raising)
      - "-NONE-ABAR-OTHER-" (misc.)

    This function extracts a coarse family (WH / RC / OTHER) that can be used
    for diagnostics, though construction type is primarily inferred structurally.

    Args:
        none_pos: A preterminal label that starts with "-NONE-...".

    Returns:
        One of {"WH", "RC", "OTHER"}.
    """
    if "ABAR-WH" in none_pos:
        return "WH"
    if "ABAR-RC" in none_pos:
        return "RC"
    return "OTHER"

# -----------------------------
# New label scheme helpers
# -----------------------------

VERB_TAGS = {"VB", "VBD", "VBP", "VBZ", "VBN", "VBG"}

def is_wh_phrase(t: ParentedTree) -> bool:
    return base_cat(t.label()).startswith("WH")

def subtree_has_wh(scope: ParentedTree) -> bool:
    return any(is_wh_phrase(st) for st in scope.subtrees())

def subtree_has_word(scope: ParentedTree, w: str) -> bool:
    w = w.lower()
    return any(isinstance(x, str) and x.lower() == w for x in scope.leaves())

def nearest_scope_sbar_or_sbarq(node: ParentedTree) -> Optional[ParentedTree]:
    return nearest_ancestor(node, lambda x: base_cat(x.label()) in {"SBARQ", "SBAR"})

def is_matrix_scope(scope: ParentedTree) -> bool:
    # MATRIX if SBARQ directly under ROOT (your earlier heuristic)
    return (
        base_cat(scope.label()) == "SBARQ"
        and scope.parent() is not None
        and base_cat(scope.parent().label()) == "ROOT"
    )

def is_np_relative_scope(scope: ParentedTree) -> bool:
    # REL_CLAUSE if SBAR directly under NP
    return (
        base_cat(scope.label()) == "SBAR"
        and scope.parent() is not None
        and base_cat(scope.parent().label()) == "NP"
    )

def find_embedding_verb_for_scope(scope: ParentedTree) -> Optional[str]:
    """
    Heuristic: For an embedded question SBAR/SBARQ, find the closest *embedding* VP
    that dominates this scope, then pick the nearest verb *in that VP* that precedes
    the SBAR/SBARQ.
    Returns surface form (lowercased), e.g. "tell", "wonder".
    """
    if scope.parent() is None:
        return None

    # Find a VP ancestor such that `scope` is inside it and VP likely "selects" it.
    vp = nearest_ancestor(scope, lambda x: base_cat(x.label()) == "VP")
    if vp is None:
        return None

    # Find the child of `vp` that contains the scope (direct or indirect)
    # We'll use treepositions to compare linear precedence within VP.
    scope_pos = scope.treeposition()

    # Collect verb preterminals in the VP
    verb_preterms = []
    for st in vp.subtrees(lambda x: isinstance(x, ParentedTree) and x.height() == 2):
        pos = base_cat(st.label())
        if pos in VERB_TAGS:
            child = st[0]
            if isinstance(child, str):
                verb_preterms.append(st)

    if not verb_preterms:
        return None

    # Prefer a verb that *precedes* the SBAR/SBARQ (common in complements: V ... SBAR)
    candidates = []
    for vpt in verb_preterms:
        vpos = vpt.treeposition()
        if vpos < scope_pos:
            candidates.append(vpt)

    # If none precede, fall back to the first verb in VP
    chosen = candidates[-1] if candidates else verb_preterms[0]
    return str(chosen[0]).lower()


# -----------------------------
# Polar detection (no WH movement)
# -----------------------------
def detect_matrix_polar(root: ParentedTree) -> bool:
    """
    Heuristic: matrix yes/no question often has SBARQ under ROOT but no WH phrase.
    """
    for st in root:
        if isinstance(st, ParentedTree) and base_cat(st.label()) == "SBARQ":
            return not subtree_has_wh(st)
    return False

def detect_embedded_polar(scope: ParentedTree) -> bool:
    """
    Heuristic: embedded polar often uses "whether" or "if" inside SBAR.
    """
    if base_cat(scope.label()) != "SBAR":
        return False
    return subtree_has_word(scope, "whether") or subtree_has_word(scope, "if")


# -----------------------------
# New construction typing -> MQ/EQ/RC
# -----------------------------
def construction_bucket_for_trace(trace_preterm: ParentedTree, filler: Optional[ParentedTree]) -> Optional[str]:
    """
    Return one of {"MQ","EQ","RC"} for WH-trace dependencies, else None.
    """
    if filler is None or not base_cat(filler.label()).startswith("WH"):
        return None

    scope = nearest_scope_sbar_or_sbarq(trace_preterm)
    if scope is None:
        return None

    if is_matrix_scope(scope):
        return "MQ"
    if is_np_relative_scope(scope):
        return "RC"
    return "EQ"


# -----------------------------
# Gap role mapping -> SUBJ / OBJ / ADJUNCT (+ POSS for RC)
# -----------------------------
def gap_role_bucket(gap_phrase: ParentedTree) -> str:
    """
    Collapse everything into {"SUBJ","OBJ","ADJUNCT"} with:
      - POBJ -> OBJ
      - PP/ADJP/ADVP/etc -> ADJUNCT
    """
    gb = base_cat(gap_phrase.label())
    parent = gap_phrase.parent()
    pb = base_cat(parent.label()) if parent is not None else None

    # Clause subject heuristic
    if gb == "NP" and pb in {"S", "SQ", "SINV"} and parent is not None:
        phrasal_children = [c for c in parent if isinstance(c, ParentedTree)]
        first_np = next((c for c in phrasal_children if base_cat(c.label()) == "NP"), None)
        if first_np is gap_phrase:
            return "SUBJ"

    # Objects
    if gb == "NP" and pb == "VP":
        return "OBJ"
    if gb == "NP" and pb == "PP":
        return "OBJ"  # previous POBJ -> OBJ

    # Everything else -> adjunct bucket
    return "ADJUNCT"


def is_possessive_rc(filler: Optional[ParentedTree]) -> bool:
    """
    RC_POSS: heuristically, WH phrase contains 'whose'.
    """
    if filler is None:
        return False
    if not base_cat(filler.label()).startswith("WH"):
        return False
    return any(isinstance(w, str) and w.lower() == "whose" for w in filler.leaves())


# -----------------------------
# Revised extract_dependencies
# -----------------------------
def extract_dependencies(tree_str: str) -> List[Dict[str, Any]]:
    """
    Now returns only labels in:
      MQ_SUBJ, MQ_OBJ, MQ_ADJUNCT, MQ_POLAR,
      EQ_SUBJ, EQ_OBJ, EQ_ADJUNCT, EQ_POLAR,
      RC_SUBJ, RC_OBJ, RC_ADJUNCT, RC_POSS

    Also adds:
      - embed_verb for EQ_* labels (best-effort, can be None)
    """
    root = ParentedTree.fromstring(tree_str)
    deps: List[Dict[str, Any]] = []

    # ---- 1) Polar questions (no trace needed)
    # Matrix polar
    if detect_matrix_polar(root):
        deps.append({
            "idx": None,
            "none_pos": None,
            "tracekind": None,
            "family": "POLAR",
            "gap_phrase_label": None,
            "gap_phrase_base": None,
            "gap_role": None,
            "gap_subtype": "POLAR",
            "filler_label": None,
            "filler_base": None,
            "construction": "MQ",
            "final_label": "MQ_POLAR",
            "embed_verb": None,
        })

    # Embedded polar: find SBARs that contain whether/if and look like complements
    for sbar in root.subtrees(lambda x: isinstance(x, ParentedTree) and base_cat(x.label()) == "SBAR"):
        if detect_embedded_polar(sbar):
            ev = find_embedding_verb_for_scope(sbar)
            deps.append({
                "idx": None,
                "none_pos": None,
                "tracekind": None,
                "family": "POLAR",
                "gap_phrase_label": None,
                "gap_phrase_base": None,
                "gap_role": None,
                "gap_subtype": "POLAR",
                "filler_label": None,
                "filler_base": None,
                "construction": "EQ",
                "final_label": "EQ_POLAR",
                "embed_verb": ev,
            })

    # ---- 2) WH-trace dependencies (your original core)
    for preterm in root.subtrees(lambda x: x.height() == 2):
        pos = preterm.label()
        if not pos.startswith("-NONE-"):
            continue

        word = preterm[0]
        if not isinstance(word, str):
            continue

        m = TRACE_WORD_RE.match(word)
        if not m:
            continue

        idx = int(m.group("idx"))
        gap_phrase = preterm.parent()

        filler = find_filler_by_index(root, idx) or find_wh_in_scope(preterm)

        family = trace_family(pos)  # WH/RC/OTHER coarse (kept for debugging)
        bucket = construction_bucket_for_trace(preterm, filler)
        if bucket is None:
            # ignore OTHER_ labels entirely as requested
            continue

        # RC possessive special-case
        if bucket == "RC" and is_possessive_rc(filler):
            final_label = "RC_POSS"
            gap_subtype = "POSS"
            gap_role = "POSS"
        else:
            gap_subtype = gap_role_bucket(gap_phrase)  # SUBJ/OBJ/ADJUNCT
            gap_role = gap_subtype
            final_label = f"{bucket}_{gap_subtype}"

        embed_verb = None
        if bucket == "EQ":
            scope = nearest_scope_sbar_or_sbarq(preterm)
            if scope is not None:
                embed_verb = find_embedding_verb_for_scope(scope)

        deps.append({
            "idx": idx,
            "none_pos": pos,
            "tracekind": m.group("tracekind"),
            "family": family,

            "gap_phrase_label": gap_phrase.label() if gap_phrase is not None else None,
            "gap_phrase_base": base_cat(gap_phrase.label()) if gap_phrase is not None else None,
            "gap_role": gap_role,
            "gap_subtype": gap_subtype,

            "filler_label": filler.label() if filler else None,
            "filler_base": base_cat(filler.label()) if filler else None,

            "construction": bucket,          # "MQ"/"EQ"/"RC"
            "final_label": final_label,      # e.g. "EQ_OBJ"
            "embed_verb": embed_verb,        # only meaningful for EQ_*
        })

    return deps




# -----------------------------
# Main: build and write outputs
# -----------------------------
def process_file(in_path: Path, out_prefix: Optional[str] = None) -> Tuple[Path, Path, Path]:
    """
    Process a `.parsed` file containing many bracketed trees and write CSV outputs.

    For each tree in the input file:
      - extract the surface sentence
      - extract and classify all trace-based dependencies
      - assign a stable tree_id (0-based index by file order)

    Output files:
      1) <prefix>_sentences.csv
         Columns: tree_id, sentence, has_trace, n_deps

      2) <prefix>_dependencies.csv
         One row per dependency (trace). Includes sentence, filler/gap labels, and final_label.

      3) <prefix>_tree_final_classification.csv
         One row per tree. Adds `final_labels_str`, a semicolon-separated set of unique
         final_labels found in that tree (or "NONE" if no dependencies).

    Args:
        in_path: Path to the input `.parsed` file.
        out_prefix: Prefix for output filenames. If None, uses `in_path.stem`.

    Returns:
        A tuple of Paths: (sentences_csv, dependencies_csv, tree_final_csv).

    Side effects:
        Writes CSV files to the same directory as `in_path`.

    Notes:
        - The tree_id is deterministic for a given file (based on file order).
        - Trees may contain multiple trace dependencies; tree_final aggregates them.
    """
    if out_prefix is None:
        out_prefix = in_path.stem

    tree_rows: List[Dict[str, Any]] = []
    dep_rows: List[Dict[str, Any]] = []

    for tree_id, ts in enumerate(iter_tree_strings(in_path)):
        sent = extract_sentence(ts)
        deps = extract_dependencies(ts)

        tree_rows.append({
            "tree_id": tree_id,
            "sentence": sent,
            "has_trace": bool(deps),
            "n_deps": len(deps), # number of dependencies (traces) in this tree
        })

        for dep_id, d in enumerate(deps):
            dep_rows.append({
                "tree_id": tree_id,
                "dep_id": dep_id,
                "sentence": sent,
                **d,
            })

    trees_df = pd.DataFrame(tree_rows)
    deps_df = pd.DataFrame(dep_rows)

    # Tree-level aggregation of final labels
    if len(deps_df) > 0:
        agg = (deps_df.groupby("tree_id")["final_label"]
                      .apply(lambda xs: ";".join(sorted(set(xs))))
                      .reset_index(name="final_labels_str"))
    else:
        agg = pd.DataFrame({"tree_id": [], "final_labels_str": []})

    trees_final = trees_df.merge(agg, on="tree_id", how="left")
    trees_final["final_labels_str"] = trees_final["final_labels_str"].fillna("NONE")

    # Write files
    out_sent = in_path.with_name(f"{out_prefix}_sentences.csv")
    out_deps = in_path.with_name(f"{out_prefix}_dependencies.csv")
    out_treefinal = in_path.with_name(f"{out_prefix}_tree_final_classification.csv")

    trees_df.to_csv(out_sent, index=False)
    deps_df.to_csv(out_deps, index=False)
    trees_final.to_csv(out_treefinal, index=False)

    # Quick summary print
    print(f"Wrote:\n  {out_sent}\n  {out_deps}\n  {out_treefinal}")
    print(f"Trees: {len(trees_df)} | Trees w/ trace: {int(trees_df['has_trace'].sum())} | Deps: {len(deps_df)}")
    # if len(deps_df) > 0:
    #     print("\nTop final labels:")
    #     print(deps_df["final_label"].value_counts().head(20))

    return out_sent, out_deps, out_treefinal
