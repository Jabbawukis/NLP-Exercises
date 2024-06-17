"""
These test cases do not cover all cases. Passing the tests does not guarantee a perfect solution!

We are using max_n=2 here for illustrative purposes. Your functions should work with any number.
"""

import pytest
from bleu_score import brevity_penalty, count_lengths_get_precision, corpus_bleu_score, get_ngram_overlap


bp_test_data = [
    (5, 4, 1),
    (0, 2, 0),
    (3, 7,  0.263597),
    (7, 11, 0.564718),
]

@pytest.mark.parametrize("candidate_length, reference_length, expected_penalty", bp_test_data)
def test_brevity_penalty(candidate_length, reference_length, expected_penalty):
    assert brevity_penalty(candidate_length, reference_length) == pytest.approx(expected_penalty, abs=1e-5)


test_candidates = [
    "what up ?", # 3 tokens; 1 1-gram match, 0 2-gram matches
    "hurry up",  # 2 tokens; 2 1-gram matches, 1 2-gram match
    "sure sure", # 2 tokens, 2 (clipped to 1) 1-gram matches, 0 2-gram matches
]

test_references_list = [
    ["!", "what 's that ?", "what is it ?"],
    ["hurry up .", "hurry it up !"],
    ["Are you sure ?", "Are you sure about this ?"],
]

def test_count_lengths_get_precision():
    precision_scores, total_candidate_length, total_reference_length = count_lengths_get_precision(test_candidates, test_references_list, 2)
    assert total_candidate_length == 7
    assert total_reference_length == 11  # 4 for the first candidate (best match), 3 for the second, and 4 for the third
    
    assert precision_scores == [
        (5/7),  # 5 matches total, 3 1-grams in first, 2 1-grams each in second and third candidate (i.e. 7 total)
        (1/4),  # 1 match total ("hurry up"), 2 2-grams in first, 1 2-gram in second and third candidate 
    ]

@pytest.mark.parametrize("precision_scores, expected", [([5/7, 1/4], 0.422577), ([0, 1], 0)])
def test_get_ngram_overlap(precision_scores, expected):
    assert get_ngram_overlap(precision_scores, 2) == pytest.approx(expected)


def test_corpus_bleu_score():
    assert corpus_bleu_score(test_candidates, test_references_list, 2) == pytest.approx(
        0.564718 * 0.422577
    )
