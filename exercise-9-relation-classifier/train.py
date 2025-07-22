from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
import wandb
import os

# Load dataset
dataset = load_from_disk("conll_04")

run = "test_6"

output_dir = f"./results/{run}"

wandb.login(key="")
os.environ["WANDB_PROJECT"] = "relation_classifier"

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-small", num_labels=2)

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]"], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))


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


for split in dataset:
    dataset[split] = dataset[split].map(process, batched=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=20,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    report_to="wandb",
    run_name=run,
    warmup_ratio=0.1
)

data_collator = DataCollatorWithPadding(tokenizer)

accuracy = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
trainer.save_model(output_dir)
wandb.finish()
