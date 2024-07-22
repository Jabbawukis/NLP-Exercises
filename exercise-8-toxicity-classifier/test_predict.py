import pytest

from predict import Classifier


@pytest.fixture(scope="session")
def classifier():
    return Classifier(model_path="/results/test_2")


@pytest.mark.parametrize(
    "sentence,true_label",
    [
        ("That's great.", "non-toxic"),
        ("Someone should stomp her face into a puddle of mush....", "toxic"),
    ],
)
def test_classifier(classifier, sentence, true_label):
    label = classifier.predict(sentence)

    assert label == true_label
