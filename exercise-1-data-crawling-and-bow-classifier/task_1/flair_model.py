from typing import Dict, List
from flair.data import Sentence
from flair.models import TextClassifier


def predict_labels(publisher_headlines: Dict[str, list[str]]) -> Dict[str, list[list[str]]]:
    # You can return a dictionary with the following structure, but you can adjust it as needed.:
    # {
    #     "publisher_1": ["positive", "negative", ..., "positive"],
    #     "publisher_2": ["positive", "negative", ..., "positive"]
    # }
    sentiments_per_publisher = {}

    # 1) Load the pre-trained flair model
    # 2) Iterate over the headlines of each publisher and convert each headline to a Sentence object
    # 3) Predict the sentiment for each headline
    # 4) Store the predicted sentiment in the dictionary

    model = TextClassifier.load('sentiment')
    for publisher, headlines in publisher_headlines.items():
        sentiments_per_publisher[publisher] = []
        for headline in headlines:
            sentence = Sentence(headline)
            model.predict(sentence)
            sentiments_per_publisher[publisher].append([headline, sentence.get_label().value])

    return sentiments_per_publisher
