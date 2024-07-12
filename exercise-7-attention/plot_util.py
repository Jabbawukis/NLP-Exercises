from typing import List, Optional
from task_1.helper_functions import *

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention(
    attention_values: torch.Tensor,
    source_tokens: Optional[List[str]] = None,
    target_tokens: Optional[List[str]] = None,
):
    if source_tokens is not None and not attention_values.shape[0] == len(source_tokens)+1:
        msg = f"The number of source tokens {len(source_tokens)+1} does not match the size of the attention_values ({attention_values.shape})."
        raise ValueError(msg)

    fig, ax = plt.subplots()
    ax.imshow(attention_values.T, cmap="cividis")

    if target_tokens is not None:
        ax.set_xticks(torch.arange(len(target_tokens)), labels=target_tokens)

    if source_tokens is not None:
        ax.set_yticks(torch.arange(len(source_tokens)), labels=source_tokens)

    ax.set_xlabel("Target tokens", c="gray", fontsize=8)
    ax.set_ylabel("Source tokens", c="gray", fontsize=8)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(attention_values.shape[0]):
        for j in range(attention_values.shape[1]):
            v = attention_values[i, j].item()
            ax.text(
                i, j, f"{v:.02f}", ha="center", va="center", color="k"
            )

    fig.tight_layout()
    return fig


if __name__ == "__main__":

    model = torch.load("task_1/best_model_attention.pt")

    # Example source and target sentences
    source_sentence = "Hallo Antoni .".lower().split()
    target_sentence = "Hi Antoni .".lower().split()

    source_onehots = make_encoder_onehot_vectors([source_sentence], model.source_dictionary, 'cpu')
    target_inputs, target_targets = make_decoder_onehot_vectors([target_sentence], model.target_dictionary, 'cpu')

    # Forward pass through the model
    encoder_outputs, encoder_hidden = model.encode(source_onehots)
    _, attention_distributions = model.decode_attention(target_inputs, encoder_outputs, encoder_hidden)

    # Visualize the attention
    attention_matrix = torch.stack(attention_distributions, dim=1).squeeze().squeeze().detach().numpy()

    plot_attention(attention_matrix, source_tokens=source_sentence, target_tokens=target_sentence)
    plt.savefig("attention_plots/plot_1.png")
    plt.show()
