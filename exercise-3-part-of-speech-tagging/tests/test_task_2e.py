import torch

from task_2.tagger import FixedContextWordTagger


def test_fixed_context_tagger():
    tagger = FixedContextWordTagger(
        vocab_size=5, num_tags=25, embedding_dim=5, context_size=2
    )

    assert (
        tagger.embedding.num_embeddings == 5
    ), "You must have changed the initializer."
    assert tagger.embedding.embedding_dim == 5, "You must have changed the initializer."

    assert tagger.linear.in_features == 5 * 5, "You must have changed the initializer."
    assert tagger.linear.out_features == 25, "You must have changed the initializer."

    assert isinstance(
        tagger.forward(list(range(5))), dict
    ), "The tagger should return a dictionary."
    assert isinstance(
        tagger.forward(list(range(5)), list(range(5))), dict
    ), "The tagger should return a dictionary."

    tagger.embedding.weight.data = torch.eye(5)
    tagger.linear.weight.data = torch.eye(25)
    tagger.linear.bias.data = torch.arange(25) * 0.01

    all_log_probs = tagger.forward(list(range(5)))["log_probs"]
    assert all_log_probs.shape == (5, 25), "The log probs have incorrect dimensions."

    log_probs = all_log_probs[2]

    x = torch.arange(25) * 0.01
    for i in range(5):
        x[i * 6] += 1.0

    x = x.unsqueeze(0)

    assert torch.allclose(
        torch.nn.functional.log_softmax(x, dim=-1), log_probs
    ), "The context is not implemented correctly."
