from task_2.train import prepare_train_sample


def test_prepare_train_sample():
    sample_tokens = ["X", "A", "B", "X", "C"]
    sample_pos_tags = ["a", "b", "a", "b", "a"]
    expected_values = ([0, 1, 2, 0, 3], [0, 1, 0, 1, 0])
    sample_vocab = {"<UNK>": 0, "A": 1, "B": 2, "C": 3}
    sample_tag_dictionary = {"a": 0, "b": 1}

    assert (
        prepare_train_sample(
            sample_tokens,
            sample_pos_tags,
            unk_token="<UNK>",
            vocab=sample_vocab,
            tag_dictionary=sample_tag_dictionary,
        )
        == expected_values
    )
