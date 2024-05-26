import random
from typing import List

import torch


def generate_piece(prob_recurse) -> List[str]:
    if random.random() > prob_recurse:
        return  ["(", ")"]

    elif random.random() < 0.5:
        return ["("] + generate_piece(prob_recurse=prob_recurse**2) + [")"]

    else:
        return generate_piece(prob_recurse=prob_recurse**2) + generate_piece(
            prob_recurse=prob_recurse**3
        )


def generate_sample(random_corruption: bool = True):
    sequence = generate_piece(0.99)

    if random_corruption and random.random() < 0.5:
        i = random.randint(0, len(sequence) - 1)
        if sequence[i] == ")":
            sequence[i] = "("
        else:
            sequence[i] = ")"

        return sequence, False

    return sequence, True


token_dict = {"(": 0, ")": 1}


def tokens_to_ids(tokens: List[str]):
    return torch.tensor(
        [token_dict[token] for token in tokens if token in token_dict]
    ).unsqueeze(0)
