import argparse
import json
from pathlib import Path

import torch

from task_2.tagger import FixedContextWordTagger, SingleWordTagger
from task_2.train import prepare_train_sample


def cli_loop(path: Path):
    if not path.exists():
        print(f"Path '{path}' not found.")
        quit(2)
    elif path.is_file():
        # hopefully a file in a result directory was passed
        path = path.parent

    # Load the stats file which contains the configuration as well as the accuracy
    with open(path / "stats.json") as f:
        stats = json.load(f)

    # Load the model checkpoint
    checkpoint = torch.load(path / "checkpoint.pt")

    # Store references to the vocabulary, tag dictionary, and configuration
    vocab = stats["vocab"]
    config = stats["config"]
    tag_dictionary = stats["tag_dictionary"]

    # Create a reverse tag dictionary to decode predicted tag ids
    reverse_tag_dictionary = {tag_id: tag for tag, tag_id in tag_dictionary.items()}

    # Instanciate the right tagger class
    if config["context_size"] is None:
        tagger = SingleWordTagger(
            vocab_size=len(vocab),
            num_tags=len(tag_dictionary),
            embedding_dim=config["embedding_dim"],
        )
    else:
        tagger = FixedContextWordTagger(
            vocab_size=len(vocab),
            num_tags=len(tag_dictionary),
            embedding_dim=config["embedding_dim"],
            context_size=config["context_size"],
        )

    # Load the model checkpoint
    tagger.load_state_dict(checkpoint)

    # Move the tagger into the evaluation setting (not really necessary, since we are not using dropout)
    tagger.eval()

    # Print out some basic information on the tagger
    print(
        f"Loaded {tagger.__class__.__name__} trained with the following configuration:"
    )
    for k, v in config.items():
        print(f"- {k}: {v}")

    print()

    print(
        f"On the validation set, it achieved a performance of {stats['accuracy']:.02%}. Here, you can try it out "
        f"yourself."
    )
    print()

    # Print out some basic information on how to use this script
    print(
        "To end the interactive loop, issue an end-of-file (EOF; Control-d in unix systems)."
    )
    print("Entered tokens should be separated by whitespace.")
    print()

    while True:  # Keep iterating until we encounter an EOF or some other interrupt
        try:
            # Read in a line
            line = input("Tokens: ")

            # Split up the tokens: since we pass to argument here, python will split at any whitespace
            tokens = line.split()

            # Convert the tokens into the ids (based on the vocabulary)
            token_ids, _ = prepare_train_sample(
                [
                    t.lower() for t in tokens
                ],  # Since the tagger was trained only on lowercase, we will do the same here
                [],  # We have no ground truth to encode here
                unk_token=tagger.UNK_TOKEN,
                tag_dictionary=tag_dictionary,
                vocab=vocab,
            )

            # Retrieve the tag id with the highes log probability fo each token
            outputs = tagger(token_ids)
            predictions = outputs["log_probs"].argmax(-1)

            # Print out the token together with the predicted tag
            print(
                " ".join(
                    f"{token} [{reverse_tag_dictionary[tag_id.item()]}]"
                    for token, tag_id in zip(tokens, predictions)
                )
            )
        except EOFError:
            print()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=Path,
        help="path to results directory which contains the model",
    )

    args = parser.parse_args()

    cli_loop(path=args.results_path)
