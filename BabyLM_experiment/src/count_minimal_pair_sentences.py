#!/usr/bin/env python3

import csv
from pathlib import Path


def build_counts():
    counts = {
        "nogap_animate_embedded": 0,
        "gap_animate_embedded": 0,
        "nogap_animate_matrix": 0,
        "gap_animate_matrix": 0,
        "nogap_inanimate_matrix": 0,
        "gap_inanimate_matrix": 0,
        "nogap_inanimate_embedded": 0,
        "gap_inanimate_embedded": 0,
        "intransitive": 0,
        "gap_subj_embedded": 0,
        "nogap_subj_embedded": 0,
        "gap_subj_matrix": 0,
        "nogap_subj_matrix": 0,
        "gap_relative": 0,
        "nogap_relative": 0,
        "gap_animate_matrix_strict": 0,
        "nogap_animate_matrix_strict": 0,
        "gap_inanimate_matrix_strict": 0,
        "nogap_inanimate_matrix_strict": 0,
        "gap_animate_matrix_qmark": 0,
        "nogap_animate_matrix_period": 0,
        "gap_inanimate_matrix_qmark": 0,
        "nogap_inanimate_matrix_period": 0,
    }

    intransitive_verbs = [
        "yawned", "fell", "sneezed", "arrived", "screamed", "cried", "disappeared", "laughed", "rose", "came"
    ]
    animate = ["told", "fed", "paid", "promised", "invited", "rescued"]
    inanimate_embedded = [
        "made", "wanted", "cut", "built", "wrote", "enjoyed", "bought", "mentioned", "fixed", "posted",
        "threw", "opened", "ordered", "pulled", "sold", "wore", "caught", "filled", "tied", "tasted",
        "locked", "recorded", "loaded", "grabbed", "delivered", "packed", "cleaned"
    ]
    inanimate_with_objects = [
        ["made", "a cake"], ["got", "the gift"], ["felt", "the cloth"], ["wanted", "the gift"], ["cut", "the cake"],
        ["tried", "the dish"], ["built", "the tower"], ["wrote", "the book"], ["enjoyed", "the show"], ["bought", "the gift"],
        ["fixed", "the bike"], ["posted", "the picture"], ["realized", "the dream"], ["threw", "the ball"], ["opened", "the door"],
        ["spread", "the news"], ["ordered", "the package"], ["understood", "the problem"], ["pulled", "the rope"], ["sold", "the book"],
        ["wore", "the clothes"], ["caught", "the ball"], ["changed", "the color"], ["tied", "the rope"], ["filled", "the bucket"],
        ["noticed", "the flower"], ["dropped", "the ball"], ["tasted", "the food"], ["locked", "the door"], ["split", "the apple"],
        ["connected", "the dots"], ["passed", "the test"], ["recorded", "the song"], ["remembered", "the song"], ["loaded", "the car"],
        ["grabbed", "the keys"], ["delivered", "the package"], ["packed", "the suitcase"], ["cleaned", "the closet"]
    ]
    animate_present = [
        "call", "interview", "fire", "protect", "thank", "arrest", "bless", "scare", "destroy", "kiss", "hug", "join",
        "remind", "help", "question", "tell", "bother", "surprise", "frighten"
    ]
    inanimate_present = [
        "make", "get", "feel", "want", "cut", "try", "build", "write", "enjoy", "bring", "mention", "fix", "post", "realize",
        "throw", "open", "spread", "order", "understand", "pull", "sell", "wear", "catch", "change", "fill", "tie", "notice",
        "drop", "taste", "lock", "split", "connect", "pass", "record", "remember", "load", "grab", "deliver", "pack", "clean"
    ]
    animate_present_strict = [
        "interview", "fire", "thank", "arrest", "kiss", "hug", "protect", "surprise", "frighten", "remind", "follow", "blame",
        "forgive", "teach", "visit"
    ]
    inanimate_present_strict = [
        "build", "write", "enjoy", "mention", "fix", "post", "throw", "open", "order", "sell", "wear", "catch", "fill", "tie",
        "drop", "lock", "grab", "deliver", "pack", "clean", "find", "lose", "hide", "break"
    ]
    embedded_question_verbs = [
        "discovered", "forgot", "knew", "remembered", "saw", "noticed", "realized", "understood", "learned", "guessed"
    ]
    factive_verbs = ["knew", "realized", "noticed", "saw", "discovered", "remembered", "forgot"]
    nouns = [
        "you", "I", "we", "they", "he", "she", "it", "the doctor", "the person", "the singer", "the teacher", "the student",
        "the parent", "the child", "the artist", "the friend", "John", "Mary", "Alex", "the neighbor", "Catherine", "the astronaut"
    ]
    animate_nouns = [
        "you", "I", "we", "they", "he", "she", "the doctor", "the person", "the singer", "the teacher", "the student", "the parent",
        "the child", "the artist", "the friend", "John", "Mary", "Alex", "the neighbor", "Catherine", "the astronaut"
    ]
    objects = [
        "you", "me", "us", "them", "him", "her", "it", "the doctor", "the person", "the singer", "the teacher", "the student",
        "the parent", "the child", "the artist", "the friend", "John", "Mary", "Alex", "the neighbor", "Catherine", "the astronaut"
    ]
    animate_objects = [
        "you", "me", "us", "them", "him", "her", "the doctor", "the person", "the singer", "the teacher", "the student", "the parent",
        "the child", "the artist", "the friend", "John", "Mary", "Alex", "the neighbor", "Catherine", "the astronaut"
    ]

    for noun in animate_nouns:
        for _helper in embedded_question_verbs:
            for noun2 in animate_nouns:
                if noun2 != noun:
                    for _verb in animate:
                        counts["nogap_animate_embedded"] += 1
                        counts["gap_animate_embedded"] += 1

    for noun in animate_nouns:
        for _helper in embedded_question_verbs:
            for noun2 in animate_nouns:
                if noun2 != noun:
                    for _verb in inanimate_embedded:
                        counts["nogap_inanimate_embedded"] += 1
                        counts["gap_inanimate_embedded"] += 1

    for _noun in nouns:
        for _verb in animate_present:
            counts["nogap_animate_matrix"] += 1
            counts["gap_animate_matrix"] += 1

    for _noun in nouns:
        for _verb in inanimate_present:
            counts["nogap_inanimate_matrix"] += 1
            counts["gap_inanimate_matrix"] += 1

    for _noun in nouns:
        for _verb in animate_present_strict:
            counts["nogap_animate_matrix_strict"] += 1
            counts["gap_animate_matrix_strict"] += 1

    for _noun in nouns:
        for _verb in inanimate_present_strict:
            counts["nogap_inanimate_matrix_strict"] += 1
            counts["gap_inanimate_matrix_strict"] += 1

    for _noun in nouns:
        for _verb in animate_present:
            counts["gap_animate_matrix_qmark"] += 1
            counts["nogap_animate_matrix_period"] += 1

    for _noun in nouns:
        for _verb in inanimate_present:
            counts["gap_inanimate_matrix_qmark"] += 1
            counts["nogap_inanimate_matrix_period"] += 1

    for noun in animate_nouns:
        for _helper in factive_verbs:
            for noun2 in animate_nouns:
                if noun2 != noun:
                    for _verb in intransitive_verbs:
                        counts["intransitive"] += 1

    for noun in animate_nouns:
        for _helper in embedded_question_verbs:
            for noun2 in animate_objects:
                if noun2 != noun:
                    for _verb in animate:
                        counts["gap_subj_embedded"] += 1
                        counts["nogap_subj_embedded"] += 1

    for _noun2 in objects:
        for _verb in animate_present:
            counts["gap_subj_matrix"] += 1
            counts["nogap_subj_matrix"] += 1

    for noun in animate_nouns:
        for noun2 in animate_nouns:
            if noun != noun2:
                for _pair in inanimate_with_objects:
                    counts["gap_relative"] += 1
                    counts["nogap_relative"] += 1

    sentence_dict_counts = {
        "nogap_animate_embedded": counts["nogap_animate_embedded"],
        "gap_animate_embedded": counts["gap_animate_embedded"],
        "nogap_animate_matrix": counts["nogap_animate_matrix"],
        "gap_animate_matrix": counts["gap_animate_matrix"],
        "nogap_inanimate_matrix": counts["nogap_inanimate_matrix"],
        "gap_inanimate_matrix": counts["gap_inanimate_matrix"],
        "nogap_inanimate_embedded": counts["nogap_inanimate_embedded"],
        "gap_inanimate_embedded": counts["gap_inanimate_embedded"],
        "intransitive": counts["intransitive"],
        "gap_subj_embedded": counts["gap_subj_embedded"],
        "nogap_subj_embedded": counts["nogap_subj_embedded"],
        "gap_subj_matrix": counts["gap_subj_matrix"],
        "nogap_subj_matrix": counts["nogap_subj_matrix"],
        "gap_relative": counts["gap_relative"],
        "nogap_relative": counts["nogap_relative"],
        "animate_embedded": counts["gap_animate_embedded"] + counts["nogap_animate_embedded"],
        "inanimate_embedded": counts["gap_inanimate_embedded"] + counts["nogap_inanimate_embedded"],
        "animate_matrix": counts["gap_animate_matrix"] + counts["nogap_animate_matrix"],
        "inanimate_matrix": counts["gap_inanimate_matrix"] + counts["nogap_inanimate_matrix"],
        "subj_embedded": counts["gap_subj_embedded"] + counts["nogap_subj_embedded"],
        "subj_matrix": counts["gap_subj_matrix"] + counts["nogap_subj_matrix"],
        "embedded": (
            counts["gap_animate_embedded"]
            + counts["gap_inanimate_embedded"]
            + counts["gap_subj_embedded"]
            + counts["nogap_animate_embedded"]
            + counts["nogap_inanimate_embedded"]
            + counts["nogap_subj_embedded"]
        ),
        "matrix": (
            counts["gap_animate_matrix"]
            + counts["gap_inanimate_matrix"]
            + counts["gap_subj_matrix"]
            + counts["nogap_animate_matrix"]
            + counts["nogap_inanimate_matrix"]
            + counts["nogap_subj_matrix"]
        ),
        "relative": counts["gap_relative"] + counts["nogap_relative"],
        "animate_matrix_strict": counts["gap_animate_matrix_strict"] + counts["nogap_animate_matrix_strict"],
        "inanimate_matrix_strict": counts["gap_inanimate_matrix_strict"] + counts["nogap_inanimate_matrix_strict"],
        "matrix_strict": (
            counts["gap_animate_matrix_strict"]
            + counts["gap_inanimate_matrix_strict"]
            + counts["nogap_animate_matrix_strict"]
            + counts["nogap_inanimate_matrix_strict"]
        ),
        "animate_matrix_qmark": counts["gap_animate_matrix_qmark"],
        "animate_matrix_period": counts["nogap_animate_matrix_period"],
        "inanimate_matrix_qmark": counts["gap_inanimate_matrix_qmark"],
        "inanimate_matrix_period": counts["nogap_inanimate_matrix_period"],
        "matrix_continuation": (
            counts["gap_animate_matrix_qmark"]
            + counts["nogap_animate_matrix_period"]
            + counts["gap_inanimate_matrix_qmark"]
            + counts["nogap_inanimate_matrix_period"]
        ),
    }

    return sentence_dict_counts


def write_csv(counts, output_path):
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["minimal_pair_category", "sentence_count"])
        for category, count in counts.items():
            writer.writerow([category, count])


def main():
    output_path = Path("figures/minimal_pair_counts.csv")
    counts = build_counts()
    write_csv(counts, output_path)
    print(f"Wrote {len(counts)} categories to {output_path}")


if __name__ == "__main__":
    main()