import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from task_2.data_util import (
    get_free_path,
    make_tag_dictionary,
    make_vocabulary,
    read_pos_from_file,
)
from task_2.plotting import plot_results
from task_2.tagger import (
    FixedContextWordTagger,
    SingleWordTagger,
    tagger_with_pretrained_embeddings,
)
from task_2.train import evaluate, train


def train_and_evaluate(
    learning_rate: float,
    embedding_dim: int,
    epochs: int,
    pretrained_embedding: Optional[Path],
    context_size: Optional[int],
    seed: int,
):
    # Set seed for reproducibility
    results_path = get_free_path("results")

    torch.manual_seed(seed)

    train_data = read_pos_from_file("data/train_data.txt")
    test_data = read_pos_from_file("data/validation_data.txt")

    tag_dictionary = make_tag_dictionary(train_data)

    if context_size is None:
        if pretrained_embedding:
            model, vocab = tagger_with_pretrained_embeddings(
                SingleWordTagger,
                "data/glove_filtered_50d.txt",
                num_tags=len(tag_dictionary),
            )
        else:
            vocab = make_vocabulary(train_data, unk_token=SingleWordTagger.UNK_TOKEN)
            model = SingleWordTagger(
                vocab_size=len(vocab),
                num_tags=len(tag_dictionary),
                embedding_dim=embedding_dim,
            )

    else:
        if pretrained_embedding:
            model, vocab = tagger_with_pretrained_embeddings(
                FixedContextWordTagger,
                "data/glove_filtered_50d.txt",
                num_tags=len(tag_dictionary),
                context_size=context_size,
            )
        else:
            vocab = make_vocabulary(
                train_data, unk_token=FixedContextWordTagger.UNK_TOKEN
            )
            model = FixedContextWordTagger(
                vocab_size=len(vocab),
                num_tags=len(tag_dictionary),
                embedding_dim=embedding_dim,
                context_size=context_size,
            )

    results = {
        "tag_dictionary": tag_dictionary,
        "config": {
            "learning_rate": learning_rate,
            "embedding_dim": embedding_dim,
            "epochs": epochs,
            "pretrained_embedding": pretrained_embedding,
            "context_size": context_size,
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
        default=0.01,
        type=float,
        help="learning rate used to train the tagger",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed used to initialize parameters"
    )
    parser.add_argument(
        "--embedding_dim",
        default=100,
        type=int,
        help="output dimension of the embedding layer",
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of epochs to train for"
    )
    parser.add_argument(
        "--pretrained_embedding",
        action="store_true",
        help="use this flag to load pretrained embedding vectors",
    )
    parser.add_argument(
        "--context_size",
        default=None,
        type=int,
        help="if specified (with an int) the FixedContextWordTagger is used with the specified context size (e.g. "
             "context size of 2 means two tokens to the left and to the right of the word to be token to be tagged "
             "are used in the prediction)",
    )

    args = parser.parse_args()

    train_and_evaluate(
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        pretrained_embedding=args.pretrained_embedding,
        context_size=args.context_size,
        seed=args.seed,
    )
