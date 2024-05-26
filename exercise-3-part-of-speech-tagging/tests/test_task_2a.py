from task_2.data_util import read_pos_from_file


def test_reading_training_data():
    data = read_pos_from_file("../data/train_data.txt")

    assert isinstance(data, list)
    assert isinstance(data[0], tuple)
    assert len(data[0]) == 2
    assert data[0][0] == ["so", "what", "happened", "?"]
    assert data[0][1] == ["RB", "WP", "VBD", "."]
