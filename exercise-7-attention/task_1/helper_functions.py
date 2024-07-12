from typing import Dict
import torch

def make_dictionary(data, unk_threshold: int = 0) -> Dict[str, int]:
    '''
    Makes a dictionary of words given a list of tokenized sentences.
    :param data: List of (sentence, label) tuples
    :param unk_threshold: All words below this count threshold are excluded from dictionary and replaced with UNK
    :return: A dictionary of string keys and index values
    '''

    # First count the frequency of each distinct ngram
    word_frequencies = {}
    for sent in data:
        for word in sent:
            if word not in word_frequencies:
                word_frequencies[word] = 0
            word_frequencies[word] += 1

    # Assign indices to each distinct ngram
    word_to_ix = {'<PAD>': 0, '<UNK>': 1, '<SEP>': 2, '<STOP>': 3}
    for word, freq in word_frequencies.items():
        if freq > unk_threshold:  # only add words that are above threshold
            word_to_ix[word] = len(word_to_ix)

    # Print some info on dictionary size
    print(f"At unk_threshold={unk_threshold}, the dictionary contains {len(word_to_ix)} words")
    return word_to_ix


def make_decoder_onehot_vectors(sentences, word_to_ix, device):

    encoded = make_encoder_onehot_vectors(sentences, word_to_ix, device, decode=True)
    target_inputs = encoded[:, :-1]
    target_targets = encoded[:, 1:]

    return target_inputs, target_targets


def make_encoder_onehot_vectors(sentences, word_to_ix, device, decode=False):
    onehot_mini_batch = []

    lengths = [len(sentence) for sentence in sentences]
    longest_sequence_in_batch = max(lengths)

    for sentence in sentences:

        if decode:
            onehot_for_sentence = [word_to_ix["<SEP>"]]
        else:
            onehot_for_sentence = []

        # move a window over the text
        for word in sentence:

            # look up ngram index in dictionary
            if word in word_to_ix:
                onehot_for_sentence.append(word_to_ix[word])
            else:
                onehot_for_sentence.append(word_to_ix["<UNK>"] if "<UNK>" in word_to_ix else 0)

        if decode:
            # append a STOP index
            onehot_for_sentence.append(word_to_ix["<STOP>"])

        # fill the rest with PAD indices
        for i in range(longest_sequence_in_batch - len(sentence)):
            onehot_for_sentence.append(word_to_ix["<PAD>"])

        onehot_mini_batch.append(onehot_for_sentence)

    onehot_mini_batch = torch.tensor(onehot_mini_batch).to(device)

    return onehot_mini_batch
