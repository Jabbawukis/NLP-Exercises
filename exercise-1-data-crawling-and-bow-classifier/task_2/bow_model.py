import torch
import torch.nn.functional as F


# Define your model
class BoWClassifier(torch.nn.Module):  # inherits from nn.Module!

    def __init__(self, vocab_size, num_labels):
        # calls the init function of nn.Module.  Don't get confused by syntax, just always do it in an nn.Module
        super(BoWClassifier, self).__init__()
        self.linear = torch.nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        features = self.linear(bow_vec)
        return F.log_softmax(features, dim=1)


def make_word_dictionary(data) -> {}:
    # word_to_ix maps each word in the vocab to a unique integer, which will be its
    # index into the Bag of words vector
    word_to_ix = {}
    for sent, _ in data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    word_to_ix["<UNK>"] = len(word_to_ix)
    return word_to_ix


def make_label_dictionary(data) -> {}:
    # word_to_ix maps each word in the vocab to a unique integer, which will be its
    # index into the Bag of words vector
    label_to_ix = {}
    for _, label in data:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
    return label_to_ix


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        try:
            vec[word_to_ix[word]] += 1
        except KeyError:
            vec[word_to_ix["<UNK>"]] += 1
    return vec.view(1, -1)


def make_label_vector(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])
