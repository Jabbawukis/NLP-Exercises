import sys
from pathlib import Path

from task_1.data import generate_sample, tokens_to_ids
from task_1.model import RNNModel

WEIGHTS_PATH = Path("task_1") / "weights.txt"
TOTAL_EVALUATIONS = 1000


def show_examples(n=20):
    for _ in range(n):
        sequence, label = generate_sample()

        print(" ".join(sequence), label)


def print_step(token, state):
    print(f"'{token}' → [RNN] →- {' '.join(f'{v:.2f}' for v in state.tolist())}")
    print(" " * 8 + "↓")


def print_prediction(logit, prediction, label):
    print("  ", "[Classifier]")
    print(" " * 8 + "↓")

    print(f"{logit:>11.2f}")
    print(" " * 8 + "↓")
    print("    ", "[Sigmoid]")
    print(" " * 8 + "↓")

    pred = prediction >= 0.5

    if label is not None:
        print(
            f"{prediction:>11.2f}",
            ("> 0.5 (is part)" if pred else "< 0.5 (is not part)"),
            ":",
            "correct" if pred == label else "wrong",
        )

    else:
        print(
            f"{prediction:>11.2f}",
            ("> 0.5 (is part)" if pred else "< 0.5 (is not part)"),
        )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in (
        "examples",
        "evaluate",
        "single",
        "interactive",
    ):
        print(f"Usage: {sys.argv[0]} <examples|single|evaluate|interactive>")
        quit(1)

    if sys.argv[1] == "examples":
        show_examples()
        quit(0)

    model = RNNModel()

    if WEIGHTS_PATH.exists():
        model.load_weights(WEIGHTS_PATH)

    model.eval()

    if sys.argv[1] == "single":
        sequence, label = generate_sample()

        seq_ids = tokens_to_ids(sequence)

        print("".join(sequence), ":", label)
        print()

        output = model(seq_ids)

        for token, hidden_state in zip(sequence, output["outputs"][0]):
            print_step(token, hidden_state)

        print_prediction(
            logit=output["logit"].item(),
            prediction=output["prediction"].item(),
            label=label,
        )

    elif sys.argv[1] == "interactive":
        state_dict = model.forward_sequence(tokens_to_ids([]))

        LINE_UP = "\033[1A"
        LINE_CLEAR = "\x1b[2K"

        while True:
            try:
                tokens = list(
                    "".join(input("Next tokens (leave empty to predict): ").split())
                )  # remove all whitespaces
                print(LINE_UP, end=LINE_CLEAR)

                if len(tokens) == 0:
                    break

                token_ids = tokens_to_ids(tokens)
                state_dict = model.forward_sequence(token_ids, state_dict)

                for token, hidden_state in zip(tokens, state_dict["outputs"][0]):
                    print_step(token, hidden_state)

            except EOFError:
                break

        state_dict = model.forward_classify(state_dict)

        print_prediction(
            logit=state_dict["logit"].item(),
            prediction=state_dict["prediction"].item(),
            label=None,
        )

    else:
        num_correct = 0
        for i in range(TOTAL_EVALUATIONS):
            sequence, label = generate_sample()
            seq_ids = tokens_to_ids(sequence)

            output = model(seq_ids)

            pred = output["prediction"].item() >= 0.5

            if pred == label:
                num_correct += 1

        acc = num_correct / TOTAL_EVALUATIONS
        print(f"Accuracy: {acc:.2%}")
