from flair.data import Sentence
from flair.models import TextClassifier


def main():
    # make a sentence
    sentence = Sentence("I love Berlin .")

    # load the NER tagger
    tagger = TextClassifier.load("sentiment")

    # run NER over sentence
    tagger.predict(sentence)

    # print the sentence with all annotations
    print(sentence)


if __name__ == "__main__":
    main()
