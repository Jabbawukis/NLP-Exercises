from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

DataFormat = List[Tuple[List[str], List[str]]]


def read_pos_from_file(path) -> DataFormat:
    """
    Read data from a file and return a list of tuples (list of tokens, list of pos tags).

    :param path: Path to the file
    """
    # [
    #     ([token_0, token_1, token_2, ...], [tag_0, tag_1, tag_2, ...]),
    #     ...
    # ]
    #
    # The tokens as well as the tags should remain strings for now

    data: DataFormat = []
    tokens = []
    tags = []
    with open(path, "r") as file:
        for line in file:
            if line == "\n":
                data.append((tokens, tags))
                tokens = []
                tags = []
            else:
                tokens.append(line.strip().split(" ")[0])
                tags.append(line.strip().split(" ")[1])
    return data


def load_token_embeddings(path) -> Tuple[List[str], torch.Tensor]:
    data = np.loadtxt(path, dtype="str", comments=None)
    tokens = list(data[:, 0])
    vectors = data[:, 1:].astype("float")

    return tokens, torch.tensor(vectors.astype("float32"))


def save_token_embeddings(path, tokens: List[str], vectors: torch.Tensor):
    vector_data = vectors.numpy()
    token_data = np.array(tokens)

    data = np.concatenate(
        (token_data.reshape((-1, 1)), vector_data.astype("str")), axis=-1, dtype="str"
    )
    np.savetxt(path, data, fmt="%s")


def make_vocabulary(data: DataFormat, *, unk_token: Optional[str] = "<UNK>"):
    """
    Make a vocab dictionary from the training data.

    :param token: List of list of tokens
    :return: A dictionary  of string keys and index values
    """
    token_to_idx: Dict[str, int] = {}
    for sent, _ in data:
        for token in sent:
            if token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)

    if unk_token:
        token_to_idx["<UNK>"] = len(token_to_idx)
    return token_to_idx


def make_tag_dictionary(data: DataFormat) -> Dict[str, int]:
    """
    Make a dictionary of pos tags.
    :param data: List of tuples (lists of words, list of pos tags)
    :return: A dictionary of string keys and index values
    """
    tag_to_idx: Dict[str, int] = {}
    for _, tags in data:
        for tag in tags:
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
    return tag_to_idx


def get_free_path(base_path: str = "results") -> Path:
    """Get an unoccupied path in the results directory to save training artefacts."""
    for i in range(10000):
        path = Path(base_path) / f"{i:04}"

        if path.exists():
            continue

        path.mkdir(parents=True)
        return path

    msg = "You should take a break!"
    raise RuntimeError(msg)
