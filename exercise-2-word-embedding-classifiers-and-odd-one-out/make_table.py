import json
import os
from prettytable import PrettyTable

# Function to extract required information from a JSON file
def extract_info(file_path):
    with open(file_path) as f:
        data = json.load(f)
        learning_rate = data.get("learning_rate")
        max_ngrams = data.get("max_ngrams")
        unk_threshold = data.get("unk_threshold")
        num_epochs = data.get("num_epochs")
        hidden_size = data.get("hidden_size")
        pretrained_embedding = data.get("pretrained_word_embeddings_file")
        test_accuracy = data.get("test_accuracy")
        test_run = f.name.split("_")[-1].split(".")[0]

        return learning_rate, max_ngrams, unk_threshold, num_epochs, hidden_size, pretrained_embedding, test_accuracy, test_run

# List to store extracted information
results = []

# Iterate through each file
for file_name in os.listdir("results"):
    if file_name.endswith(".json"):
        file_path = os.path.join("results", file_name)
        info = extract_info(file_path)
        results.append(info)

# Create PrettyTable
table = PrettyTable(["Learning Rate", "N-Grams", "Unk Threshold", "Num Epochs", "Hidden Size", "Pretrained Embedding",
                     "Accuracy on Test Set", "Run"])

# Add rows to the table
for result in sorted(results, key=lambda x: int(x[-1])):  # Sort by the "Run" column
    row = list(result)
    for i, val in enumerate(row):
        if i == len(row) - 2 and val > 0.80:
            row[i] = f"****{val}****"  # Highlight accuracy over 80
    table.add_row(row)

# Set formatting options
table.align = "l"
table.padding_width = 1

# Print or save the table
with open("results_task_1.txt", "w") as f:
    f.write(str(table))
