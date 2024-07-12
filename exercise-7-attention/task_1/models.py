import math
from typing import Dict

import torch
import torch.nn.functional as F

from task_1.helper_functions import make_encoder_onehot_vectors, make_decoder_onehot_vectors


class Attention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, encoder_outputs, decoder_hidden):
        att_score = torch.matmul(decoder_hidden, encoder_outputs.transpose(2, 1))
        attention_distribution = self.softmax(att_score)
        attention_output = torch.matmul(attention_distribution, encoder_outputs)
        return (
            attention_output,  # step 3 in attention calculation: [batch_size x 1 x hidden_size]
            attention_distribution  # step 2 in attention calculation: [batch_size x 1 x source_seq_len]
        )


class Seq2Seq(torch.nn.Module):

    def __init__(self,
                 source_dictionary: Dict[str, int],
                 target_dictionary: Dict[str, int],
                 embedding_size: int = 256,
                 lstm_hidden_size: int = 512,
                 device: str = 'cpu',
                 num_layers: int = 1,
                 use_attention: bool = False
                 ):

        super(Seq2Seq, self).__init__()

        self.device = device
        self.source_dictionary = source_dictionary
        self.target_dictionary = target_dictionary

        self.encoder_embedding = torch.nn.Embedding(len(self.source_dictionary), embedding_size)

        self.encoder_lstm = torch.nn.LSTM(embedding_size,
                                          lstm_hidden_size,
                                          batch_first=True,
                                          num_layers=num_layers,
                                          )

        self.decoder_embedding = torch.nn.Embedding(len(self.target_dictionary), embedding_size)

        self.decoder_lstm = torch.nn.LSTM(embedding_size,
                                          lstm_hidden_size,
                                          batch_first=True,
                                          num_layers=num_layers,
                                          )
        self.use_attention = use_attention
        if use_attention:
            self.hidden2tag = torch.nn.Linear(lstm_hidden_size * 2, len(self.target_dictionary))
            self.attention = Attention(lstm_hidden_size)
        else:
            self.hidden2tag = torch.nn.Linear(lstm_hidden_size, len(self.target_dictionary))

        self.to(device)

    def encode(self, source_onehots):
        # Embed sentences in source language
        source_embeds = self.encoder_embedding(source_onehots)

        # Send through LSTM
        encoder_outputs, encoder_hidden = self.encoder_lstm(source_embeds)

        return encoder_outputs, encoder_hidden

    def decode(self, target_onehots, encoder_hidden):
        # embeds sentences in target language
        target_embeds = self.decoder_embedding(target_onehots)

        # decoder rnn: uses last encoder hidden state and generates target sequence
        decoder_out, _ = self.decoder_lstm(target_embeds, encoder_hidden)

        # prediction head
        logits = self.hidden2tag(decoder_out)

        return F.log_softmax(logits, dim=-1)

    def decode_attention(self, target_onehots, encoder_outputs, encoder_hidden):
        # Initialize list for storing outputs: all_logits should be used to calculate log_probs,
        # while attention_distributions are used for visualization (outputs from the attention computation)
        all_logits, attention_distributions = [], []
        target_embeds_batch = self.decoder_embedding(target_onehots)

        for sentence in torch.split(target_embeds_batch, 1, dim=1):
            decoder_hidden, _ = self.decoder_lstm(sentence, encoder_hidden)
            attention_output, attention_distribution = self.attention.forward(encoder_outputs, decoder_hidden)
            all_logits.append(self.hidden2tag(torch.cat([decoder_hidden, attention_output], dim=-1)))
            attention_distributions.append(attention_distribution)

        log_probs = F.log_softmax(torch.cat(all_logits, dim=1), dim=-1)
        return log_probs, attention_distributions

    def forward(self, source_onehots, target_onehots):
        # Encode sentences in source language
        encoder_outputs, encoder_hidden = self.encode(source_onehots)

        # Decode from encoder hidden state and generate outputs
        if self.use_attention:
            log_probs, _ = self.decode_attention(target_onehots, encoder_outputs, encoder_hidden)
        else:
            log_probs = self.decode(target_onehots, encoder_hidden)

        return log_probs

    def calculate_loss(self, log_probabilities_for_each_class, targets):
        # Flatten all predictions and targets for the whole mini-batch into one long list
        flattened_log_probabilities_for_each_class = log_probabilities_for_each_class.flatten(end_dim=1)
        flattened_targets = targets.flatten()

        # Calculate loss
        loss = torch.nn.functional.nll_loss(
            input=flattened_log_probabilities_for_each_class,
            target=flattened_targets,
            ignore_index=self.target_dictionary['<PAD>']
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

            return math.exp(aggregate_loss), aggregate_loss.item()
