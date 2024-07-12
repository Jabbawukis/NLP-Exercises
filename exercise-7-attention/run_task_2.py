import torch
import time
import sys

import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from task_2.models import *
from task_2.helper_funcs import *

# Which model to train and hyperparameters
model_type = sys.argv[1]
learning_rate = 0.005
number_of_epochs = 10
hidden_size = 128
batch_size = 64

# Load word embeddings
word_vectors = KeyedVectors.load_word2vec_format('task_2/glove_50d_filtered.txt', binary=False, no_header=True)

# Load dataset and split it into train/validation/test
snli_df = pd.read_csv("task_2/snli.txt", sep="\t")
train_df, val_df = train_test_split(snli_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)

train_dataset = SNLIDataset(train_df, word_vectors)
val_dataset = SNLIDataset(val_df, word_vectors)
test_dataset = SNLIDataset(test_df, word_vectors)

# Prepare dataloaders for training
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start = time.time()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        end = time.time()
        
        val_accuracy, val_loss = evaluate(model, val_loader, criterion)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model, f"task_2/best_model_{model_type}.pt")
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%, Time: {round(end - start, 2)} seconds')

def evaluate(model, data_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    average_loss = total_loss / len(data_loader)
    return accuracy, average_loss

# Define which loss to use
criterion = nn.CrossEntropyLoss()

# Train models
torch.manual_seed(1)
if model_type == "no_attention":
    model = LSTMModel(input_dim=50, hidden_dim=hidden_size, output_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Vanilla LSTM:")
elif model_type == "attention":
    model = LSTMSelfAttentionModel(input_dim=50, hidden_dim=hidden_size, output_dim=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("LSTM w/ Self-Attention:")

train(model, train_loader, val_loader, criterion, optimizer, num_epochs=number_of_epochs)
best_model = torch.load(f"task_2/best_model_{model_type}.pt")
test_result = evaluate(best_model, test_loader, criterion)
print(f'Test Accuracy: {test_result}')

with open("results.txt", "a") as outfile:
    outfile.write("Task 2" + "\t" + model_type + "\t" + str(test_result[0]) + "\n")