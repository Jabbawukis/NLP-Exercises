import torch
from task_2.model import RecurrentModel, model_with_pretrained_embeddings


def test_rnn_initialization():
    model, vocab = model_with_pretrained_embeddings(
        RecurrentModel,
        "data/glove_filtered_50d.txt",
        num_tags=25,
        hidden_dim=10,
        bidirectional=True,
    )

    assert model.embedding_dim == 50, "You must have changed the initializer."
    assert isinstance(
        model.forward(list(range(5))), dict
    ), "The forward method must return a dictionary. Have you implemented rnn_forward?"
    assert isinstance(
        model.forward(list(range(5)), list(range(5))), dict
    ), "The tagger should return a dictionary. Have you implemented rnn_forward?"


def test_rnn_unidirectional():
    torch.manual_seed(0)
    model, vocab = model_with_pretrained_embeddings(
        RecurrentModel,
        "data/glove_filtered_50d.txt",
        num_tags=5,
        hidden_dim=10,
        bidirectional=False,
    )

    assert (
        model.classifier.in_features == 10
    ), "The classifier input features are wrong. They should be the hidden size."
    assert model.bidirectional == False, "The model should be unidirectional."

    embeddings = model.embedding(torch.tensor(list(range(5))))
    hidden_states = model.rnn_forward(embeddings)
    assert hidden_states.shape == (
        5,
        10,
    ), "The hidden states after the rnn_forward have incorrect dimensions."

    all_log_probs = model.forward(list(range(5)))["log_probs"]
    assert all_log_probs.shape == (5, 5), "The log probs have incorrect dimensions."


def test_rnn_bidirectional():
    torch.manual_seed(0)
    model, vocab = model_with_pretrained_embeddings(
        RecurrentModel,
        "data/glove_filtered_50d.txt",
        num_tags=5,
        hidden_dim=10,
        bidirectional=True,
    )

    assert (
        model.classifier.in_features == 20
    ), "The classifier input features are wrong. They should be the hidden size."
    assert model.bidirectional == True, "The model should be bidirectional."

    embeddings = model.embedding(torch.tensor(list(range(5))))
    hidden_states = model.rnn_forward(embeddings)
    assert hidden_states.shape == (
        5,
        20,
    ), "The hidden states after the rnn_forward have incorrect dimensions."

    all_log_probs = model.forward(list(range(5)))["log_probs"]
    assert all_log_probs.shape == (5, 5), "The log probs have incorrect dimensions."


if __name__ == "__main__":
    test_rnn_initialization()
    test_rnn_unidirectional()
    print("Task 2 tests passed")
