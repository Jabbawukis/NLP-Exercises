from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from task_2.data_util import DataFormat
from task_2.model import RecurrentModel


def prepare_train_sample(
    tokens: List[str],
    pos_tags: List[str],
    *,
    unk_token: str,
    vocab: Dict[str, int],
    tag_dictionary: Dict[str, int],
) -> Tuple[List[int], List[int]]:

    token_ids: List[int] = [
        vocab[token] if token in vocab else vocab[unk_token] for token in tokens
    ]

    tag_ids: List[int] = [tag_dictionary[tag] for tag in pos_tags]
    return token_ids, tag_ids


def train(
    *,
    vocab: Dict[str, int],
    tag_dictionary: Dict[str, int],
    model: RecurrentModel,
    train_data: DataFormat,
    learning_rate: float,
    num_epochs: int,
    eval_freq: int = 5000,
    loss_log_freq: int = 20,
):
    # Define the loss and a simple SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize a dictionary where we will track the training dynamics
    curves = defaultdict(list)
    running_loss: float = 0.0

    step: int = -1

    for step in tqdm(range(num_epochs * len(train_data))):
        tokens, pos_tags = train_data[step % len(train_data)]

        token_ids, pos_ids = prepare_train_sample(
            tokens,
            pos_tags,
            vocab=vocab,
            tag_dictionary=tag_dictionary,
            unk_token=model.UNK_TOKEN,
        )

        # Remember that PyTorch accumulates gradients
        # We need to clear them out before each instance
        model.zero_grad()

        # Run our forward pass
        outputs: Dict = model(token_ids, pos_ids)
        loss = outputs["loss"]

        # Compute the loss, gradients, and update the parameters by calling optimizer.step()
        loss.backward()
        optimizer.step()

        if (step + 1) % loss_log_freq == 0:
            curves["running_loss"].append((step, running_loss / 100))
            running_loss = 0.0

        if (step + 1) % eval_freq == 0:
            # Evaluate the model every epoch
            eval_output = evaluate(
                model=model,
                test_data=train_data,
                vocab=vocab,
                tag_dictionary=tag_dictionary,
            )

            curves["train_accuracy"].append((step, eval_output["accuracy"]))
            curves["train_loss"].append((step, eval_output["loss"]))

    return {"curves": curves}


def evaluate(
    *,
    vocab: Dict[str, int],
    tag_dictionary: Dict[str, int],
    model: RecurrentModel,
    test_data: DataFormat,
):
    confusion_matrix: np.ndarray = np.zeros(
        (len(tag_dictionary), len(tag_dictionary)), dtype="int"
    )
    validation_loss: float = 0.0

    with torch.no_grad():
        for tokens, pos_tags in test_data:
            # Run our forward pass

            token_ids, pos_ids = prepare_train_sample(
                tokens,
                pos_tags,
                vocab=vocab,
                tag_dictionary=tag_dictionary,
                unk_token=model.UNK_TOKEN,
            )

            outputs = model(token_ids, pos_ids)

            for log_probs, tag in zip(outputs["log_probs"], pos_ids):
                prediction = torch.argmax(log_probs).item()
                confusion_matrix[tag, prediction] += 1

            # Extract the loss
            validation_loss += outputs["loss"].item()

        # Calculate the accuracy: the diagonal contains all correct predictions
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        validation_loss /= len(test_data)

        return {
            "accuracy": accuracy,
            "loss": validation_loss,
            "confusion_matrix": confusion_matrix,
        }
