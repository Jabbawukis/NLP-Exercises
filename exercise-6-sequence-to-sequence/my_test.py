import torch
import pickle
from sequence_to_sequence_model import Seq2Seq
from bleu_score import corpus_bleu_score

# Which model to evaluate
model_name = "hedgehog"
# Model state dict path
state_dict_path = f"models/state_dict-{model_name}.pt"

# Load data
data_dict = {}
with open("data/translations_sample.txt") as infile:
    for line in infile:
        source = line.split("\t")[0].strip().lower()
        target = line.split("\t")[1].strip().lower()
        if data_dict.get(source) is None:
            data_dict[source] = [target]
        else:
            if target not in data_dict[source]:
                data_dict[source].append(target)

# Load the model vocab
with open(f'models/{model_name}_source_dict.pkl', 'rb') as f:
    source_vocab = pickle.load(f)
with open(f'models/{model_name}_target_dict.pkl', 'rb') as f:
    target_vocab = pickle.load(f)

# Instantiate the seq2seq and load the state dict
translator = Seq2Seq(source_dictionary=source_vocab,
                     target_dictionary=target_vocab,
                     embedding_size=512,  # set the correct values
                     lstm_hidden_size=1024,  # set the correct values
                     num_layers=1,  # set the correct values
                     device="cuda:0"
                     )
translator.load_state_dict(torch.load(state_dict_path))

greedy_translations, sampled_translations = [], []
torch.manual_seed(1)
for source in list(data_dict.keys()):
    # do a translation once with the greedy decoding strategy, and once with the multinomial
    greedy_translations.append(translator.translate(source, decode_type="greedy", max_symbols=10))
    sampled_translations.append(translator.translate(source, decode_type="multinomial", max_symbols=10, temperature=1.))

print(corpus_bleu_score(greedy_translations, list(data_dict.values()), max_n=3))
print(corpus_bleu_score(sampled_translations, list(data_dict.values()), max_n=3))
