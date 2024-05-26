from typing import Dict, List, Optional, Tuple

import torch

from task_2.data_util import load_token_embeddings


class RecurrentModel(torch.nn.Module):
    """This is a recurrent model that takes a sequence of tokens and classifies it."""

    UNK_TOKEN = "<UNK>"

    def __init__(
            self,
            vocab_size: int,
            num_tags: int,
            embedding_dim: int,
            hidden_dim: int,
            bidirectional: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        self.forward_rnn_cell = torch.nn.RNNCell(embedding_dim, hidden_dim)
        self.backward_rnn_cell = torch.nn.RNNCell(embedding_dim, hidden_dim)

        if self.bidirectional:
            hidden_dim = hidden_dim * 2
        self.classifier = torch.nn.Linear(hidden_dim, num_tags)  # adjust so that hidden_dim *2 if bidirectional

        self.loss_function = torch.nn.NLLLoss()

    def forward(self, tokens: List[int], pos_tags: Optional[List[int]] = None):
        embeddings = self.embedding(torch.tensor(tokens))

        hidden_states = self.rnn_forward(embeddings)

        log_probs = torch.nn.functional.log_softmax(
            self.classifier(hidden_states), dim=-1
        )
        result = {"log_probs": log_probs}

        if pos_tags is not None:
            result["loss"] = self.loss_function(log_probs, torch.tensor(pos_tags))

        return result

    def rnn_forward(self, embeddings: torch.Tensor):

        if self.bidirectional:
            f_output = []
            b_output = []
            hx_f = torch.zeros(self.hidden_dim)
            hx_b = torch.zeros(self.hidden_dim)
            for embedding_forward, embedding_backward in zip(
                    embeddings, embeddings.flip(0)
            ):
                hx_f = self.forward_rnn_cell(embedding_forward, hx_f)
                hx_b = self.backward_rnn_cell(embedding_backward, hx_b)
                f_output.append(hx_f)
                b_output.append(hx_b)
            b_output.reverse()
            output = [torch.cat((f, b)) for f, b in zip(f_output, b_output)]
        else:
            f_output = []
            hx = torch.zeros(self.hidden_dim)
            for embedding in embeddings:
                hx = self.forward_rnn_cell(embedding, hx)
                f_output.append(hx)
            output = f_output

        hidden_states: torch.Tensor = torch.stack(output)

        return hidden_states


def model_with_pretrained_embeddings(
        tagger_class, path, *, add_unk: bool = True, **kwargs
) -> Tuple[RecurrentModel, Dict[str, int]]:
    """
    Load an embeddings from a path, add the unkown token, and create an instance of the recurrent word tagger.

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
