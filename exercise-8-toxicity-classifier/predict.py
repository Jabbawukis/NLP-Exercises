import sys

from transformers import pipeline


class Classifier:
    def __init__(self, model_path: str = None):
        self.classifier = pipeline("sentiment-analysis", model=model_path)

    def predict(self, text: str) -> float:
        """Predict the label given some text.

        Return the label (as a string).
        """
        label = self.classifier(text)
        return label[0]["label"]


def main():
    if len(sys.argv) < 2:
        classifier = Classifier("./results/test_2")
    else:
        classifier = Classifier(sys.argv[1])  # model path
    try:
        while True:
            text = input("> ")
            label = classifier.predict(text)
            print(f"Predicted: {label}")
    except (EOFError, KeyboardInterrupt):
        print()
        pass


if __name__ == "__main__":
    main()
