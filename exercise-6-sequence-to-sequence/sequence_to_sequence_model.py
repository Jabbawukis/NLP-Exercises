import math
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np

from helper_functions import make_encoder_onehot_vectors, make_decoder_onehot_vectors


class Seq2Seq(torch.nn.Module):

    def __init__(self,
                 source_dictionary: Dict[str, int],
                 target_dictionary: Dict[str, int],
                 embedding_size: int = 256,
                 lstm_hidden_size: int = 512,
                 device: str = 'cpu',
                 num_layers: int = 1,
                 ):

        super(Seq2Seq, self).__init__()

        self.device = device
        self.source_dictionary = source_dictionary
        self.target_dictionary = target_dictionary
        self.target_dictionary_inv = {v: k for k, v in target_dictionary.items()}

        self.encoder_embedding = torch.nn.Embedding(num_embeddings=len(source_dictionary),
                                                    embedding_dim=embedding_size)

        self.encoder_lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder_embedding = torch.nn.Embedding(num_embeddings=len(target_dictionary),
                                                    embedding_dim=embedding_size)

        self.decoder_lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.hidden2tag = torch.nn.Linear(lstm_hidden_size, len(target_dictionary))

        self.to(device)

    def encode(self, source_onehots):
        # embed sentences in source language
        source_embeds = self.encoder_embedding(source_onehots)

        # send through LSTM
        _, encoder_hidden = self.encoder_lstm(source_embeds)

        return encoder_hidden

    def decode(self, target_onehots, encoder_hidden):
        # embeds sentences in target language
        target_embeds = self.decoder_embedding(target_onehots)

        # decoder rnn: uses last encoder hidden state
        # and generates target sequence
        decoder_out, _ = self.decoder_lstm(target_embeds, encoder_hidden)

        # prediction head
        logits = self.hidden2tag(decoder_out)

        return F.log_softmax(logits, dim=2)

    def forward(self, source_onehots, target_onehots):

        # embed sentences in source language and pas through an lstm
        encoder_hidden = self.encode(source_onehots)

        # decode from encoder hidden state and generate outputs using teacher forcing
        log_probs = self.decode(target_onehots, encoder_hidden)

        return log_probs

    def calculate_loss(self, log_probabilities_for_each_class, targets):
        # flatten all predictions and targets for the whole mini-batch into one long list
        flattened_log_probabilities_for_each_class = log_probabilities_for_each_class.flatten(end_dim=1)
        flattened_targets = targets.flatten()

        # calculate loss
        loss = torch.nn.functional.nll_loss(
            input=flattened_log_probabilities_for_each_class,
            target=flattened_targets,
        )
        return loss

    def evaluate(self, test_data):
        with torch.no_grad():
            aggregate_loss = 0.

            # go through all test data points
            for sentence_pair in test_data:
                # make one-hot inputs
                source_sentences = [sentence_pair[0]]
                target_sentences = [sentence_pair[1]]

                source_inputs = make_encoder_onehot_vectors(source_sentences, self.source_dictionary, self.device)
                target_inputs, target_targets = make_decoder_onehot_vectors(target_sentences, self.target_dictionary,
                                                                            self.device)

                log_probabilities_for_each_class = self.forward(source_inputs, target_inputs)

                aggregate_loss += self.calculate_loss(log_probabilities_for_each_class, target_targets)

            # average the loss for one batch sample
            aggregate_loss = aggregate_loss / len(test_data)

            return math.exp(aggregate_loss), aggregate_loss

    def encode_step(self, sentence: str):
        source_inputs = make_encoder_onehot_vectors([sentence.split()], self.source_dictionary, self.device)
        sentence_tensor = torch.tensor(source_inputs, device=self.device)
        embedded_sentence = self.encoder_embedding(sentence_tensor)
        _, hidden_state = self.encoder_lstm(embedded_sentence)
        return hidden_state

    def decode_step(self, token: str, hidden_state):
        token_tensor = torch.tensor([[self.target_dictionary[token]]],
                                    device=self.device)
        embedded_token = self.decoder_embedding(token_tensor)
        output, hidden_state = self.decoder_lstm(embedded_token, hidden_state)
        logits = self.hidden2tag(output.squeeze())
        probabilities = F.log_softmax(logits, dim=-1).squeeze()
        probabilities[self.target_dictionary["<UNK>"]] = -np.inf
        probabilities[self.target_dictionary["<PAD>"]] = -np.inf
        probabilities[self.target_dictionary["<SEP>"]] = -np.inf

        return probabilities, hidden_state

    def translate(self, sentence: str, decode_type: str = "greedy", max_symbols: int = 10, temperature: float = 1.):
        """
        Compute the forward pass during inference.

        :param sentence: A sentence string (sentence to be translated).
        :param decode_type: Defines how to choose next token. Can be either "greedy" or "multinomial".
        :param max_symbols: The maximum amount of tokens to generate.
        :param temperature: Parameter affecting log prob distribution for the multinomial sampling strategy.
        :return: Translated string.
        """
        hidden_state = self.encode_step(sentence)
        current_token = "<SEP>"
        translation = []

        for _ in range(max_symbols):
            probabilities, hidden_state = self.decode_step(current_token, hidden_state)
            if decode_type == "greedy":
                next_token_id = torch.argmax(probabilities).item()
            elif decode_type == "multinomial":
                next_token_id = torch.multinomial(probabilities.div(temperature).exp(), num_samples=1).item()
            else:
                raise ValueError("decode_type must be either 'greedy' or 'multinomial'")
            current_token = self.target_dictionary_inv[next_token_id]
            if current_token == "<STOP>":
                break
            translation.append(current_token)

        return ' '.join(translation)
