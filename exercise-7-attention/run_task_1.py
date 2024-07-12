import sys
import time
import torch
import random
import more_itertools
import torch.optim as optim

from task_1.helper_functions import make_dictionary, make_decoder_onehot_vectors, make_encoder_onehot_vectors
from task_1.models import Seq2Seq

model_type = sys.argv[1]

# All hyperparameters
learning_rate = 0.1
number_of_epochs = 10
hidden_size = 64
emb_size = 32
mini_batch_size = 16
num_layers = 1
unk_threshold = 0
if model_type == "no_attention":
    use_attention = False
elif model_type == "attention":
    use_attention = True

print(f"Training language model with \n - hidden_size: {hidden_size}\n - learning rate: {learning_rate}"
      f" \n - max_epochs: {number_of_epochs} \n - mini_batch_size: {mini_batch_size} \n - unk_threshold: {unk_threshold}")

# dataset path
input_filename = 'task_1/name_translation.txt'

# -- Step 1: Get a small amount of training data
training_data = []
with open(input_filename) as text_file:
    for line in text_file.readlines():

        source = line.split("\t")[0]
        target = line.split("\t")[1]

        training_data.append([source.lower().split(), target.lower().split()])

corpus_size = len(training_data)
random.seed(42)
random.shuffle(training_data)

training_data = training_data[:-round(corpus_size / 5)]
validation_data = training_data[-round(corpus_size / 5):-round(corpus_size / 10)]
test_data = training_data[-round(corpus_size / 10):]

print(
    f"\nTraining corpus has {len(training_data)} train, {len(validation_data)} validation and {len(test_data)} test sentences")

all_source_sentences = [pair[0] for pair in training_data]
all_target_sentences = [pair[1] for pair in training_data]

source_dictionary = make_dictionary(all_source_sentences, unk_threshold=unk_threshold)
target_dictionary = make_dictionary(all_target_sentences, unk_threshold=unk_threshold)

device = 'cpu'
torch.manual_seed(1)
# initialize translator and send to device
model = Seq2Seq(source_dictionary=source_dictionary,
                target_dictionary=target_dictionary,
                lstm_hidden_size=hidden_size,
                embedding_size=emb_size,
                num_layers=num_layers,
                device=device,
                use_attention=use_attention
                )

# --- Do a training loop

# Define a simple SGD optimizer with a learning rate of 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Remember the best model
best_model = None
best_validation_perplexity = 100000.

# Go over the training dataset multiple times
for epoch in range(number_of_epochs):

    print(f"\n - Epoch {epoch}")

    start = time.time()

    # shuffle training data at each epoch
    random.shuffle(training_data)

    train_loss = 0.

    for batch in more_itertools.chunked(training_data, mini_batch_size):
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Run our forward pass.
        source_sentences = [pair[0] for pair in batch]
        target_sentences = [pair[1] for pair in batch]

        source_inputs = make_encoder_onehot_vectors(source_sentences, source_dictionary, device)
        target_inputs, target_targets = make_decoder_onehot_vectors(target_sentences, target_dictionary, device)

        # Run our forward pass
        log_probabilities_for_each_class = model.forward(source_inputs, target_inputs)

        # Calculate loss
        loss = model.calculate_loss(log_probabilities_for_each_class, target_targets)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(training_data)

    # Evaluate and print accuracy at end of each epoch
    validation_perplexity, validation_loss = model.evaluate(validation_data)

    # Remember best model:
    if validation_perplexity < best_validation_perplexity:
        print(f"new best model found!")
        best_validation_perplexity = validation_perplexity

        # always save best model
        torch.save(model, f"task_1/best_model_{model_type}.pt")

    # Print losses
    print(f"training loss: {train_loss}")
    print(f"validation loss: {validation_loss}")
    print(f"validation perplexity: {validation_perplexity}")

    end = time.time()
    print(f'{round(end - start, 3)} seconds for this epoch')

# Load best model and do final test
best_model = torch.load(f"task_1/best_model_{model_type}.pt")
test_result = best_model.evaluate(test_data)
print(f"Final test PPL and loss: {test_result}")

with open("results.txt", "a") as outfile:
    outfile.write("Task 1" + "\t" + model_type + "\t" + str(test_result[0]) + "\n")