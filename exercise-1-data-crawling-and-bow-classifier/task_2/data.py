from typing import Tuple, List


def get_small_training_data() -> Tuple:
    # Small dataset for testing
    training_data = [
        ("this talk is interesting".split(), "POSITIVE"),
        ("this talk is not interesting".split(), "NEGATIVE"),
        ("this idea is great".split(), "POSITIVE"),
        ("this idea is bad".split(), "NEGATIVE"),
    ]

    test_data = [
        ("this talk is great".split(), "POSITIVE"),
        ("this talk is bad".split(), "NEGATIVE"),
    ]

    return training_data, test_data


def get_large_training_data(path: str) -> List[Tuple]:
    data = []
    with open(path, "r") as file:
        for line in file:
            label, *text = line.strip().split()
            data.append((text, label.replace('__label__', '')))
    return data


def check_large_training_data(training_data, test_data):
    # some checks to make sure you loaded the data correctly
    assert len(training_data) == 20000
    assert len(test_data) == 2000
    assert training_data[0] == (["the", "greatest", "musicians"], "POSITIVE")
    assert training_data[10000] == (["strength"], "POSITIVE")
