from tqdm import tqdm

import torch
import torch.optim as optim

from task_2.bow_model import (
    make_word_dictionary,
    make_label_dictionary,
    BoWClassifier,
    make_bow_vector,
    make_label_vector,
)
from task_2.data import (
    get_small_training_data,
    get_large_training_data,
    check_large_training_data,
)
from task_2.evaluation import evaluate_final_model


def main(dataset_size):
    # Set seed for reproducibility
    torch.manual_seed(1)

    # You can choose between small and large datasets.
    if dataset_size == "small":
        training_data, test_data = get_small_training_data()
    elif dataset_size == "large":
        training_data = get_large_training_data("data/training_data.txt")
        test_data = get_large_training_data("data/validation_data.txt")
        check_large_training_data(training_data, test_data)
    else:
        raise ValueError("dataset_size should be either 'small' or 'large'")

    # -- Step 2): Get the indices for each word and label
    word_dictionary = make_word_dictionary(training_data)
    label_dictionary = make_label_dictionary(training_data)

    # (you can print the dictionaries to see what's in there)
    # print(word_dictionary)
    # print(label_dictionary)

    model = BoWClassifier(len(word_dictionary), len(label_dictionary))

    # --- Step 4: Do a training loop
    # define the loss and a simple SGD optimizer with a learning rate of 0.1
    loss_function = torch.nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Go over the training dataset 3 times (reduce this number if dataset is large)
    number_of_epochs = 10 if dataset_size == "small" else 3
    for epoch in tqdm(range(number_of_epochs), desc="Epoch", unit="epoch"):

        # go through each training data point
        for instance, label in tqdm(training_data, desc="Training", unit="data point"):

            # 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # 2. Make datapoint and its label into a vector respectively
            bow_vector = make_bow_vector(instance, word_dictionary)
            target = make_label_vector(label, label_dictionary)

            # 3. Run our forward pass.
            log_probabilities_for_each_class = model.forward(bow_vector)

            # 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(log_probabilities_for_each_class, target)
            loss.backward()
            optimizer.step()

    accuracy = evaluate_final_model(model, test_data, word_dictionary, label_dictionary)

    # Print out the accuracy
    print(
        "Accuracy for {dataset_size} dataset size: {accuracy:.2%}".format(
            dataset_size=dataset_size, accuracy=accuracy
        )
    )

    if dataset_size == "small" and accuracy < 0.9:
        print(
            "WARNING: Accuracy is too low for small dataset. Please check your implementation."
        )

    if dataset_size == "large" and accuracy < 0.5:
        print(
            "WARNING: Accuracy is too low for large dataset. Please check your implementation."
        )


if __name__ == "__main__":
    dataset_size = "large"
    main(dataset_size)
