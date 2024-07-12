import torch
import numpy as np
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SNLIDataset(Dataset):
    def __init__(self, data, word_vectors):
        self.data = data
        self.word_vectors = word_vectors
        self.label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data.iloc[idx]['sentence1']
        hypothesis = self.data.iloc[idx]['sentence2']
        label = self.data.iloc[idx]['gold_label']
        
        premise_tokens = word_tokenize(premise)
        hypothesis_tokens = word_tokenize(hypothesis)
        
        premise_embeddings = [self.word_vectors[token] if token in self.word_vectors else np.zeros(50) for token in premise_tokens]
        hypothesis_embeddings = [self.word_vectors[token] if token in self.word_vectors else np.zeros(50) for token in hypothesis_tokens]
        
        combined_embeddings = np.array(premise_embeddings + [np.ones(50)] + hypothesis_embeddings)
        combined_embeddings = torch.tensor(combined_embeddings, dtype=torch.float32)
        label_idx = self.label_map[label]
        return combined_embeddings, label_idx

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels)
    return sequences_padded, labels