from task_2.data_util import make_tag_dictionary

sample_data = [
    (("Hope", "this", "works"), ("VERB", "PRON", "VERB")),
    (("The", "car", "stopped"), ("DET", "NOUN", "VERB")),
]


def test_make_tag_dictionary():
    tag_dictionary = make_tag_dictionary(sample_data)

    assert set(tag_dictionary.keys()) == {"VERB", "PRON", "DET", "NOUN"}
    assert set(tag_dictionary.values()) == set(range(4))
