from typing import List

import math
from collections import Counter


def count_ngrams(sentence, n):
    tokens = sentence.split()
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def modified_precision(candidate, references, n):
    # Count n-grams in the candidate sentence
    candidate_ngrams = count_ngrams(candidate, n)
    max_ref_ngrams = Counter()

    # For each reference sentence, count n-grams and find the maximum count for each n-gram
    for reference in references:
        reference_ngrams = count_ngrams(reference, n)
        for ngram in candidate_ngrams:
            max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], reference_ngrams[ngram])

    # Clip the counts of candidate n-grams by the maximum counts found in the references
    clipped_counts = {ngram: min(count, max_ref_ngrams[ngram]) for ngram, count in candidate_ngrams.items()}

    # Return the sum of clipped counts and the total number of candidate n-grams
    # sum of candidate_ngrams corresponds to p_n (denominator) in the precision calculation
    return sum(clipped_counts.values()), sum(candidate_ngrams.values())


def brevity_penalty(candidate_length: int, reference_length: int):
    if candidate_length > reference_length:
        return 1
    elif candidate_length == 0:
        return 0
    else:
        return math.exp(1 - (reference_length / candidate_length))


def count_lengths_get_precision(candidates_list: List[str], references_list: List[List[str]], max_n: int):
    p_n = [0] * max_n
    counts = [0] * max_n
    precision_scores = [0.0] * max_n
    total_candidate_length, total_reference_length = 0, 0

    for candidate, references in zip(candidates_list, references_list):
        candidate_length = len(candidate.split())
        length_differences = [(abs(len(reference.split()) - candidate_length), reference)
                              for reference in references]
        best_reference = min(length_differences, key=lambda t: t[0])[1]
        reference_length = len(best_reference.split())

        total_candidate_length += candidate_length
        total_reference_length += reference_length

        for i in range(max_n):
            p, count = modified_precision(candidate, references, i + 1)
            p_n[i] += p
            counts[i] += count

    for i in range(max_n):
        precision_scores[i] = p_n[i] / counts[i] if counts[i] > 0 else 0

    return precision_scores, total_candidate_length, total_reference_length


def get_ngram_overlap(precision_scores: List[float], max_n: int):
    if any(p == 0 for p in precision_scores):
        return 0
    score = sum(math.log(p) for p in precision_scores) / max_n
    return math.exp(score)


def corpus_bleu_score(candidates_list, references_list, max_n=4):
    ### Task 2a
    # Iterate over each candidate and its corresponding reference sentences, and 
    # calculate precision for each n-gram size
    precision_scores, total_candidate_length, total_reference_length = count_lengths_get_precision(
        candidates_list, references_list, max_n)

    ### Task 2b
    # Calculate the geometric mean of the precisions for each n-gram size 
    ngram_overlap = get_ngram_overlap(precision_scores, max_n)

    ### Task 2c
    # Calculate the brevity penalty
    bp = brevity_penalty(total_candidate_length, total_reference_length)

    # Return the final BLEU score
    return bp * ngram_overlap
