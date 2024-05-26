from typing import Dict, List, Optional, Tuple, Union

import torch

from task_2.data_util import load_token_embeddings


class SingleWordTagger(torch.nn.Module):
    """The single word tagger assigns pos tags only based on the individualy token."""

    UNK_TOKEN = "<UNK>"

    def __init__(self, vocab_size: int, num_tags: int, embedding_dim: int):
        """Initialize the SingleWordTagger.

        :param vocab_size: Size of the vocabulary
        :param num_tags: Number of tags that can be preddicted
        :param embedding_dim: Output size of the embedding layer
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.loss_function = torch.nn.NLLLoss()

        self.linear = torch.nn.Linear(embedding_dim, num_tags)

    def forward(self, tokens: List[int], pos_tags: Optional[List[int]] = None):
        log_probs_list = []

        for token in tokens:
            embedding = self.embedding(torch.tensor([token]))
            features = self.linear(embedding)
            log_probs_list.append(torch.nn.functional.log_softmax(features, dim=-1))

        log_probs = torch.cat(log_probs_list, dim=0)

        result = {"log_probs": log_probs}

        if pos_tags is not None:
            result["loss"] = self.loss_function(log_probs, torch.tensor(pos_tags))

        # Return the loss and the log probabilities
        # return {"loss": <loss: float>, "log_probs": <log probs for all tokens: torch.Tensor>}
        return result


class FixedContextWordTagger(torch.nn.Module):
    """The fixed context word tagger assigns tags based on a context window which extends `context_size` tokens left and right from the respective token.


    <token_0> ... <token_(i - context_size)> ... <token_i> ... <token_(i+context_size)> ... <token_n>
                  |-------------------------------------------------------------------|
                            relevant tokens to predict POS tag for token i


    """

    UNK_TOKEN = "<UNK>"

    def __init__(
            self, vocab_size: int, num_tags: int, embedding_dim: int, context_size: int
    ):
        """Initialize the FixedContextWordTagger.

        :param vocab_size: Size of the vocabulary
        :param num_tags: Number of tags that can be preddicted
        :param embedding_dim: Output size of the embedding layer
        :param context_size: Number of tokens to take into account when predicting a POS tag.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.loss_function = torch.nn.NLLLoss()

        self.context_size = context_size

        # The input for this layer is the concatenation of all relevant tokens (i.e. the token itself + context_size tokens to the left + context_size tokens to the right)
        self.linear = torch.nn.Linear(embedding_dim * (1 + 2 * context_size), num_tags)

    def forward(self, tokens: List[int], pos_tags: Optional[List[int]] = None):

        embeddings = [self.embedding(torch.tensor([token])) for token in tokens]
        log_probs_list = []
        for idx, embedding in enumerate(embeddings):
            list_of_embeddings = []
            if idx - self.context_size < 0:
                left_padding = [torch.zeros_like(embedding)] * (self.context_size - idx)
                list_of_embeddings.extend(left_padding)
                list_of_embeddings.extend(embeddings[:idx + 1])
            else:
                list_of_embeddings.extend(embeddings[idx - self.context_size:idx + 1])
            if idx + self.context_size >= len(embeddings):
                right_padding = [torch.zeros_like(embedding)] * (self.context_size - (len(embeddings) - idx - 1))
                list_of_embeddings.extend(embeddings[idx + 1:])
                list_of_embeddings.extend(right_padding)
            else:
                list_of_embeddings.extend(embeddings[idx + 1:idx + self.context_size + 1])

            cat_embeddings = torch.cat(list_of_embeddings, dim=1)
            features = self.linear(cat_embeddings)
            log_probs_list.append(torch.nn.functional.log_softmax(features, dim=-1))

        log_probs: torch.Tensor = torch.cat(log_probs_list, dim=0)

        assert log_probs.shape == (len(tokens), self.num_tags)

        result = {"log_probs": log_probs}

        if pos_tags is not None:
            result["loss"] = self.loss_function(log_probs, torch.tensor(pos_tags))

        return result


def tagger_with_pretrained_embeddings(
        tagger_class, path, *, add_unk: bool = True, **kwargs
) -> Tuple[Union["SingleWordTagger", "FixedContextWordTagger"], Dict[str, int]]:
    """
    Load an embeddings from a path, add the unkown token, and create an instance of the word tagger.

    :param path: Path to load the embedding from
    :param add_unk: Whether to add a <UNK> token
    :param **kwargs: Other keyword args that will be passed to the initializer
    :return: Tuple containing the tagger and the vocabulary
    """
    tokens, embedding_data = load_token_embeddings(path)

    # Create the vocabulary dictionary from the list of loaded tokens
    vocabulary = {token: i for i, token in enumerate(tokens)}

    # Add the <UNK> token if it does not exist
    if add_unk and tagger_class.UNK_TOKEN not in vocabulary:
        # Initialize the unk token as the average of all other tokens
        embedding_data = torch.cat(
            (embedding_data, embedding_data.mean(dim=0).unsqueeze(0))
        )
        vocabulary[tagger_class.UNK_TOKEN] = len(vocabulary)

    # Create the model
    model = tagger_class(
        vocab_size=len(vocabulary), embedding_dim=embedding_data.size(-1), **kwargs
    )

    # Set the embedding vectors to the loaded data
    model.embedding.weight.data = embedding_data

    return model, vocabulary
