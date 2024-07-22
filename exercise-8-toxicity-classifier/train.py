from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import numpy as np
import wandb
import os

output_dir = "./results/"

wandb.login(key="")
os.environ["WANDB_PROJECT"] = "toxicity_sentiment_analysis"

model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny",
    num_labels=2,
    label2id={"non-toxic": 0, "toxic": 1},
    id2label={0: "non-toxic", 1: "toxic"},
)

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

dataset = load_dataset("HU-Berlin-ML-Internal/toxicity-dataset")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="wandb"
)

data_collator = DataCollatorWithPadding(tokenizer)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(output_dir)
wandb.finish()
