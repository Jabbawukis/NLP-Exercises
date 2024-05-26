import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from task_2.data_util import (
    get_free_path,
    make_tag_dictionary,
    read_pos_from_file,
)
from task_2.model import RecurrentModel, model_with_pretrained_embeddings
from task_2.train import evaluate, train
from task_2.plotting import plot_results


def train_and_evaluate(
    learning_rate: float,
    hidden_dim: int,
    bidirectional: bool,
    epochs: int,
    seed: int,
):
    # Set seed for reproducibility
    results_path = get_free_path("results")

    torch.manual_seed(seed)

    train_data = read_pos_from_file("data/train_data.txt")
    test_data = read_pos_from_file("data/validation_data.txt")

    tag_dictionary = make_tag_dictionary(train_data)
    model, vocab = model_with_pretrained_embeddings(
        RecurrentModel,
        "data/glove_filtered_50d.txt",
        num_tags=len(tag_dictionary),
        hidden_dim=hidden_dim,
        bidirectional=bidirectional,
    )

    results = {
        "tag_dictionary": tag_dictionary,
        "config": {
            "learning_rate": learning_rate,
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "epochs": epochs,
            "seed": seed,
        },
        "vocab": vocab,
    }

    train_results = train(
        model=model,
        vocab=vocab,
        tag_dictionary=tag_dictionary,
        learning_rate=learning_rate,
        num_epochs=epochs,
        train_data=train_data,
    )

    results.update(train_results)

    eval_results = evaluate(
        vocab=vocab, tag_dictionary=tag_dictionary, model=model, test_data=test_data
    )

    results.update(eval_results)

    torch.save(model.state_dict(), results_path / "checkpoint.pt")

    print("Final Accuracy:", results["accuracy"])

    print("Saving results at:", results_path)
    with open(results_path / "stats.json", "w") as f:
        json.dump(
            {
                **results,
                "confusion_matrix": results["confusion_matrix"].tolist(),
            },
            f,
        )

    plot_results(results, results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        default=0.1,
        type=float,
        help="learning rate used to train the tagger",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed used to initialize parameters"
    )
    parser.add_argument(
        "--hidden_dim",
        default=100,
        type=int,
        help="the hidden dimension of the recurrent cell",
    )
    parser.add_argument(
        "--epochs", default=20, type=int, help="number of epochs to train for"
    )
    parser.add_argument(
        "--bidirectional", action="store_true", help="use a bidirectional RNN"
    )
    args = parser.parse_args()

    train_and_evaluate(
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        epochs=args.epochs,
        seed=args.seed,
    )
