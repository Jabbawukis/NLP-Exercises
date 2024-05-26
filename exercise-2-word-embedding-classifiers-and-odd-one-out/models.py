from abc import abstractmethod
from typing import Optional, List
from tqdm import tqdm

import torch
import torch.nn.functional as F

from gensim.models.keyedvectors import Word2VecKeyedVectors


class Classifier(torch.nn.Module):
    """Abstract class for a classifier. You need to implement the forward method.
    Evaluation method is shared of all classifiers."""

    def __init__(
            self,
            vocab_size: int,
            num_labels: int,
            loss_function: torch.nn.Module = torch.nn.NLLLoss,
    ):
        super(Classifier, self).__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.loss_function = loss_function()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def evaluate(
            self,
            test_data: List,
    ) -> float:
        tp: int = 0
        fp: int = 0

        validation_loss: float = 0.0

        # Create an iterator over the training data
        data_iterator = iter(test_data)
        evaluation_steps = len(test_data)
        pbar = tqdm(range(1, evaluation_steps + 1))

        with torch.no_grad():
            for evaluation_step in pbar:

                # Get next instance and label
                vector, label = next(data_iterator)

                # Run our forward pass
                outputs = self.forward(vector, label)

                # Extract the log probabilities
                log_probs = outputs["log_probs"]

                # Check if the predicted label is correct
                if torch.argmax(log_probs).item() == label.item():
                    tp += 1
                else:
                    fp += 1

                # Extract the loss
                validation_loss += outputs["loss"]

                # Update the progress bar
                description = f"Evaluation | {evaluation_step}/{len(test_data)}"
                pbar.set_description(description)

            # Calculate the accuracy
            accuracy = tp / (tp + fp)
            validation_loss /= len(test_data)

            return {"accuracy": accuracy, "validation_loss": validation_loss}


class WordEmbeddingClassifier(Classifier):
    """This is a simple FastText / WordEmbedding classifier."""

    def __init__(
            self,
            vocab_size: int,
            num_labels: int,
            hidden_size: int,
            pretrained_embeddings: Optional[Word2VecKeyedVectors] = None,
    ):
        super().__init__(vocab_size=vocab_size, num_labels=num_labels)
        self.hidden_size = hidden_size
        if not pretrained_embeddings:
            self.embedding = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings.vectors), freeze=False)
            # adding one new dimension for UNK Token
            self.embedding.weight = torch.nn.Parameter(
                torch.cat((self.embedding.weight, torch.randn(1, self.hidden_size))))
            # Adjusted the embedding size according to
            # https://discuss.pytorch.org/t/expanding-pretrained-embedding/83370 for missing UNK in
            # pretrained embeddings
        self.linear = torch.nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, one_hot_sentence, target):
        # Return the dictionary with the loss and the log probabilities
        embeddings = self.embedding.forward(one_hot_sentence)
        pooled_embeddings = torch.mean(embeddings, dim=1)
        features = self.linear(pooled_embeddings)
        log_probs = F.log_softmax(features, dim=1)
        loss = self.loss_function(log_probs, target)
        return {"loss": loss, "log_probs": log_probs}


class BoWClassifier(Classifier):
    """This is a simple Bag-of-Words classifier. You know this class from the previous exercise."""

    def __init__(self, vocab_size: int, num_labels: int):
        super().__init__(vocab_size=vocab_size, num_labels=num_labels)
        self.linear = torch.nn.Linear(self.vocab_size, self.num_labels)

    def forward(self, bow_vec, target):
        features = self.linear(bow_vec)
        log_probs = F.log_softmax(features, dim=1)
        loss = self.loss_function(log_probs, target)
        return {"loss": loss, "log_probs": log_probs}
