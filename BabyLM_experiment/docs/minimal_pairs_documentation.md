# Minimal Pairs Documentation

This document provides detailed linguistic explanations for all minimal pair contrasts tested in `analysis.py`. These tests evaluate whether language models have acquired island constraints and fundamental syntactic knowledge about wh-movement and gap licensing.

## Table of Contents

1. [Object Extraction from Embedded Questions](#object-extraction-from-embedded-questions)
2. [Object Extraction from Matrix Questions](#object-extraction-from-matrix-questions)
3. [Subject Extraction](#subject-extraction)
4. [Relative Clauses](#relative-clauses)
5. [Intransitive Verb Constraints](#intransitive-verb-constraints)
6. [Strict Transitivity Tests](#strict-transitivity-tests)
7. [Continuation Marker Tests](#continuation-marker-tests)

---

## Object Extraction from Embedded Questions

### Animate Object Extraction (Embedded)

**Linguistic Phenomenon**: Wh-island constraint violations with animate objects in embedded questions

**Gap Condition** (`gap_animate_embedded`):
- **Good**: "I wondered who Mary told"
- **Bad**: "I wondered that Mary told"
- **Continuation**: " today"

**No-Gap Condition** (`nogap_animate_embedded`):
- **Good**: "I wondered that Mary told"
- **Bad**: "I wondered who Mary told"
- **Continuation**: " it"

**Explanation**: 
- In the **gap condition**, the wh-word "who" must license a gap (missing object). The sentence "I wondered who Mary told ___ today" is grammatical because the object position after "told" is properly empty. The alternative "I wondered that Mary told ___ today" is ungrammatical because "that" cannot license a wh-gap.
- In the **no-gap condition**, when an overt object pronoun follows ("it"), the complementizer "that" is appropriate, while "who" creates an illicit double-object structure.
- This tests whether models understand that wh-words require gaps while declarative complementizers require overt arguments.

### Inanimate Object Extraction (Embedded)

**Linguistic Phenomenon**: Wh-island constraint violations with inanimate objects in embedded questions

**Gap Condition** (`gap_inanimate_embedded`):
- **Good**: "I wondered what Mary made"
- **Bad**: "I wondered that Mary made"
- **Continuation**: " today"

**No-Gap Condition** (`nogap_inanimate_embedded`):
- **Good**: "I wondered that Mary made"
- **Bad**: "I wondered what Mary made"
- **Continuation**: " it"

**Explanation**:
- Parallel to animate object extraction, but uses inanimate wh-word "what" instead of "who".
- Tests the same constraint: wh-words ("what") require syntactic gaps, while declarative complementizers ("that") require overt objects.
- The model must predict that "what Mary made today" is more natural than "that Mary made today" when no object follows.

---

## Object Extraction from Matrix Questions

### Animate Object Extraction (Matrix)

**Linguistic Phenomenon**: Direct wh-questions with animate objects requiring gaps

**Gap Condition** (`gap_animate_matrix`):
- **Good**: "who will John call"
- **Bad**: "John will call"
- **Continuation**: " today"

**No-Gap Condition** (`nogap_animate_matrix`):
- **Good**: "John will call"
- **Bad**: "who will John call"
- **Continuation**: " someone"

**Explanation**:
- In the **gap condition**, "who will John call today" is a well-formed wh-question with a gap after "call". The alternative "John will call today" is ungrammatical as a complete sentence because the transitive verb "call" requires an object.
- In the **no-gap condition**, when an overt object follows ("someone"), the declarative "John will call someone" is grammatical, while "who will John call someone" is ungrammatical due to having both a wh-word and an overt object.
- This tests understanding of obligatory transitivity and wh-gap licensing in matrix questions.

### Inanimate Object Extraction (Matrix)

**Linguistic Phenomenon**: Direct wh-questions with inanimate objects requiring gaps

**Gap Condition** (`gap_inanimate_matrix`):
- **Good**: "what will John make"
- **Bad**: "John will make"
- **Continuation**: " today"

**No-Gap Condition** (`nogap_inanimate_matrix`):
- **Good**: "John will make"
- **Bad**: "what will John make"
- **Continuation**: " it"

**Explanation**:
- Parallel to animate matrix questions but with inanimate wh-word "what".
- Tests the same fundamental constraint: transitive verbs require objects, and wh-words must license gaps while declaratives require overt objects.

---

## Strict Transitivity Tests

These tests use **strictly transitive verbs** (verbs that cannot be used without an object, excluding ambitransitive verbs like "eat" or "read" that can be used both transitively and intransitively).

### Animate Matrix (Strict)

**Linguistic Phenomenon**: Obligatory transitivity with strictly transitive animate verbs

**Gap Condition** (`gap_animate_matrix_strict`):
- **Good**: "who will John interview"
- **Bad**: "John will interview"
- **Continuation**: " today"

**Verbs Used**: interview, fire, thank, arrest, kiss, hug, protect, surprise, frighten, remind, follow, blame, forgive, teach, visit

**Explanation**:
- These verbs are **strictly transitive** - they cannot appear without an object in grammatical English.
- "*John will interview" is completely ungrammatical without an object, making this a stronger test than using ambitransitive verbs.

### Inanimate Matrix (Strict)

**Linguistic Phenomenon**: Obligatory transitivity with strictly transitive inanimate verbs

**Gap Condition** (`gap_inanimate_matrix_strict`):
- **Good**: "what will John build"
- **Bad**: "John will build"
- **Continuation**: " today"

**Verbs Used**: build, write, enjoy, mention, fix, post, throw, open, order, sell, wear, catch, fill, tie, drop, lock, grab, deliver, pack, clean, find, lose, hide, break

**Explanation**:
- Same principle as animate strict tests, but with inanimate objects.
- Verbs like "build", "grab", "deliver" cannot be used intransitively, making the contrast clearer.

---

## Continuation Marker Tests

These tests examine whether models are sensitive to **punctuation and question markers** as cues for gap licensing.

### Animate Matrix with Question Mark

**Linguistic Phenomenon**: Question formation with question mark continuation

**Gap Condition** (`gap_animate_matrix_qmark`):
- **Good**: "who will John call?"
- **Bad**: "John will call?"
- **Continuation**: "?"

**Explanation**:
- The question mark signals that this is an interrogative sentence.
- "who will John call?" is a well-formed wh-question.
- "John will call?" without an object is ungrammatical, even with rising intonation (question mark).

### Animate Matrix with Period

**Linguistic Phenomenon**: Statement formation with period continuation

**No-Gap Condition** (`nogap_animate_matrix_period`):
- **Good**: "John will call someone."
- **Bad**: "who will John call someone."
- **Continuation**: "."

**Explanation**:
- The period signals a declarative sentence.
- "John will call someone." is a grammatical declarative with an overt object.
- "who will John call someone." is ungrammatical because wh-questions don't end with periods and the verb already has an object.

### Inanimate Matrix with Question Mark

**Gap Condition** (`gap_inanimate_matrix_qmark`):
- **Good**: "what will John build?"
- **Bad**: "John will build?"
- **Continuation**: "?"

**Explanation**: Parallel to animate question mark tests, but with inanimate objects.

### Inanimate Matrix with Period

**No-Gap Condition** (`nogap_inanimate_matrix_period`):
- **Good**: "John will build something."
- **Bad**: "what will John build something."
- **Continuation**: "."

**Explanation**: Parallel to animate period tests, but with inanimate objects.

---

## Subject Extraction

### Subject Extraction from Embedded Clauses

**Linguistic Phenomenon**: Subject extraction from embedded clauses (complex NP islands in some variants)

**Gap Condition** (`gap_subj_embedded`):
- **Good**: "I wondered who told me"
- **Bad**: "I wondered that told me"
- **Continuation**: " the story"

**No-Gap Condition** (`nogap_subj_embedded`):
- **Good**: "I wondered that they told me"
- **Bad**: "I wondered who they told me"
- **Continuation**: " the story"

**Explanation**:
- In the **gap condition**, "who" must license a subject gap. "I wondered who ___ told me" is grammatical, while "I wondered that ___ told me" is ungrammatical because "that" cannot license a missing subject.
- In the **no-gap condition**, when an overt subject "they" is present, "that" is the appropriate complementizer, while "who they" creates an ungrammatical double subject structure.

### Subject Extraction from Matrix Clauses

**Linguistic Phenomenon**: Subject wh-questions in matrix clauses

**Gap Condition** (`gap_subj_matrix`):
- **Good**: "who will call me"
- **Bad**: "will call me"
- **Continuation**: " tomorrow"

**No-Gap Condition** (`nogap_subj_matrix`):
- **Good**: "he will call me"
- **Bad**: "who will he call me"
- **Continuation**: " tomorrow"

**Explanation**:
- In the **gap condition**, "who will call me" is a grammatical subject wh-question with a gap in subject position. "will call me" is ungrammatical because English requires an overt subject.
- In the **no-gap condition**, "he will call me" is grammatical with an overt subject "he", while "who will he call me" is ungrammatical because it attempts to extract an object ("who") when the verb already has an object ("me").

---

## Relative Clauses

**Linguistic Phenomenon**: Gap vs. resumptive pronoun in relative clauses

**Gap Condition** (`gap_relative`):
- **Good**: "I saw a cake that you made yesterday"
- **Bad**: "I saw a cake that you made it yesterday"
- **Continuation**: " yesterday"

**No-Gap Condition** (`nogap_relative`):
- **Good**: "I knew that you made a cake yesterday"
- **Bad**: "I knew that you made a cake it yesterday"
- **Continuation**: " yesterday"

**Explanation**:
- In the **gap condition**, relative clauses in English require a gap, not a resumptive pronoun. "I saw a cake that you made ___" is grammatical, while "I saw a cake that you made it" is ungrammatical because English doesn't allow resumptive pronouns in standard relative clauses.
- In the **no-gap condition**, a declarative "that"-clause should have an overt object. "I knew that you made a cake" is grammatical, while "I knew that you made a cake it" is ungrammatical due to the doubled object.
- This tests whether models understand the distinction between restrictive relative clauses (which require gaps) and declarative complement clauses (which require overt arguments).

---

## Intransitive Verb Constraints

**Linguistic Phenomenon**: Subcategorization restrictions on intransitive verbs

**Test** (`intransitive`):
- **Good**: "I knew that Mary yawned today"
- **Bad**: "I knew what Mary yawned today"
- **Continuation**: " today"

**Intransitive Verbs Used**: yawned, fell, sneezed, arrived, screamed, cried, disappeared, laughed, rose, came

**Explanation**:
- Intransitive verbs like "yawn", "fall", "sneeze" do not take direct objects.
- The wh-word "what" presupposes the extraction of an object, which is incompatible with intransitive verbs.
- "I knew that Mary yawned" is grammatical because "that" introduces a simple declarative clause.
- "*I knew what Mary yawned" is ungrammatical because "what" implies there should be an object after "yawned", but intransitive verbs cannot have objects.
- This tests whether models understand verb subcategorization (which verbs can and cannot take objects).

---

## Summary of Linguistic Principles Tested

1. **Wh-Gap Dependency**: Wh-words must license syntactic gaps; they cannot co-occur with overt arguments in the same position.

2. **Obligatory Transitivity**: Transitive verbs require objects. When no wh-word is present, omitting the object is ungrammatical.

3. **Complementizer Selection**: Interrogative complementizers ("who", "what") and declarative complementizers ("that") have different licensing requirements.

4. **Subcategorization**: Verbs have restrictions on what arguments they can take (transitive vs. intransitive).

5. **Island Constraints**: Certain syntactic configurations (like some embedded questions) restrict extraction.

6. **Resumptive Pronoun Constraint**: Standard English requires gaps in relative clauses, not resumptive pronouns.

7. **Punctuation Sensitivity**: Question marks and periods provide cues about sentence type that interact with syntactic structure.

## Evaluation Method

For each minimal pair, the model is presented with:
- **Sentence A** (prompt) + **Continuation**
- **Sentence B** (prompt) + **Continuation**

The model assigns a log probability to each completion. The model is scored as **correct** if it assigns higher probability to the grammatical completion.

This measures whether the model has implicit knowledge of these syntactic constraints through distributional patterns in its training data.
