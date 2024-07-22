import os

from datasets import load_dataset
from evaluate import evaluator
from transformers import pipeline
import evaluate
from prettytable import PrettyTable

table = PrettyTable()

dataset = load_dataset("HU-Berlin-ML-Internal/toxicity-dataset", split="test").shuffle(seed=42).select(range(1000))
task_evaluator = evaluator("sentiment-analysis")

accuracies = {}

models = os.listdir("results/")
models.sort()
for model in models:
    classifier = pipeline("sentiment-analysis", model=f"results/{model}")

    eval_results = task_evaluator.compute(
        model_or_pipeline=classifier,
        data=dataset,
        metric=evaluate.combine(["accuracy", "f1"]),
        label_mapping={"non-toxic": 0, "toxic": 1}
    )
    accuracies[model] = {"accuracy": eval_results["accuracy"], "f1": eval_results["f1"]}

table.field_names = ["Model", "Accuracy", "F1"]
for key, value in accuracies.items():
    table.add_row([key, value["accuracy"], value["f1"]])

with open("scores.txt", "w") as text_file:
    text_file.write(table.get_string())
