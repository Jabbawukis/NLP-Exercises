import os

from datasets import load_from_disk
from evaluate import evaluator
from transformers import pipeline, AutoTokenizer
import evaluate
from prettytable import PrettyTable

table = PrettyTable()

dataset = load_from_disk("conll_04")["validation"]
task_evaluator = evaluator("text-classification")
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"], special_tokens=True)


def process(examples):
    examples["label"] = []
    examples["text"] = []
    for idx, token_list in enumerate(examples["tokens"]):
        encased_sentence = []
        for i, token in enumerate(token_list):
            if i == examples["subj_start"][idx]:
                encased_sentence.append("[E1]")
            elif i == examples["subj_end"][idx]:
                encased_sentence.append("[/E1]")
            elif i == examples["obj_start"][idx]:
                encased_sentence.append("[E2]")
            elif i == examples["obj_end"][idx]:
                encased_sentence.append("[/E2]")
            encased_sentence.append(token)
        is_relation = 1 if examples["relation"][idx] == "Live_In" else 0
        examples["text"].append(" ".join(encased_sentence).strip())
        examples["label"].append(is_relation)

    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(process, batched=True)

accuracies = {}

models = os.listdir("results/")
models.sort()
for model in models:
    classifier = pipeline("text-classification", model=f"results/{model}", tokenizer=tokenizer)

    eval_results = task_evaluator.compute(
        model_or_pipeline=classifier,
        data=tokenized_datasets,
        metric=evaluate.combine(["f1"]),
        label_mapping={"LABEL_1": 1, "LABEL_0": 0}
    )
    accuracies[model] = {"f1": eval_results["f1"]}

table.field_names = ["Model", "F1"]
for key, value in accuracies.items():
    table.add_row([key, value["f1"]])

with open("scores.txt", "w") as text_file:
    text_file.write(table.get_string())
