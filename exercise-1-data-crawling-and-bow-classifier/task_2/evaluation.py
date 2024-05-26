import torch
import torch.nn.functional as F

from task_2.bow_model import make_bow_vector


def evaluate_final_model(model, test_data, word_dictionary, label_dictionary) -> float:
    # evaluate the model
    correct_predictions: int = 0
    incorrect_predictions: int = 0

    with torch.no_grad():

        # go through all test data points
        for instance, label in test_data:
            # get the vector for the data point
            bow_vec = make_bow_vector(instance, word_dictionary)

            # send through model to get a prediction
            log_probs = model.forward(bow_vec)

            index_of_max_value = torch.argmax(log_probs).item()

            predicted_label = ""
            for key, value in label_dictionary.items():
                if value == index_of_max_value:
                    predicted_label = key

            if predicted_label == label:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

        return correct_predictions / (correct_predictions + incorrect_predictions)
