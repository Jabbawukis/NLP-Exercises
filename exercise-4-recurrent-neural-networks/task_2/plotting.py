from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt


def plot_curves(curves, path):
    # Create a plot
    fig, ax1 = plt.subplots()

    # Plot the curves on dual axes
    loss_curve = curves["train_loss"]
    accuracy_curve = curves["train_accuracy"]

    ax2 = ax1.twinx()
    ax1.plot(*zip(*loss_curve), "tab:blue")
    ax2.plot(*zip(*accuracy_curve), "tab:orange")

    # Set labels
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Training Accuracy")

    fig.tight_layout()

    # Save the plot
    plt.savefig(path)


def plot_confusion_matrix(cm, path, tags):
    fig, ax = plt.subplots()
    ax.imshow(cm)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(tags)), labels=tags)
    ax.set_yticks(np.arange(len(tags)), labels=tags)

    # Loop over data dimensions and create text annotations.
    for i in range(len(tags)):
        for j in range(len(tags)):
            _ = ax.text(j, i, cm[i, j], ha="center", va="center", color="w")

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    # Save the plot
    plt.savefig(path)


def plot_results(results: Dict, path: Path):
    plot_curves(results["curves"], path / "training_curve.png")
    plot_confusion_matrix(
        results["confusion_matrix"],
        path=path / "confusion_matrix.png",
        tags=results["tag_dictionary"].keys(),
    )
