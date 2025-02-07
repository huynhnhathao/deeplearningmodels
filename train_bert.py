## Importing Libraries
import torch
import os
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModel,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score
import datasets
import evaluate
from torch import nn

## Setting up the Environment
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

##Loading the Dataset
dataset = load_dataset("glue", "sst2")
print(dataset["train"][100])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

max_length = 128


def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
vali_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))


##Model Definition
class ClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super(ClassificationModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        logits = self.classifier(
            outputs.last_hidden_state[:, 0, :]
        )  # Use [CLS] token representation
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            labels = labels.type(torch.LongTensor).cuda()
            loss = loss_fct(logits, labels)
        return (loss, logits) if loss is not None else logits


##Loading the Base Model
config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2)
base_model = AutoModel.from_pretrained("bert-base-uncased", config=config)
model = ClassificationModel(base_model, num_labels=2)


##Defining Training Arguments and Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print("logits: ", logits, " labels: ", labels)
    print("eval_pred: ", eval_pred, " predictions: ", predictions)
    return {"accuracy": accuracy_score(labels, predictions)}


training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="steps",
    logging_dir="logs",
    per_device_train_batch_size=64,
    logging_steps=100,
    save_total_limit=2,  # Save only the last 2 checkpoints
    load_best_model_at_end=True,
    num_train_epochs=3,  # Increase the number of epochs if needed
    logging_first_step=True,  # Log the first step
)

##Training and evaluating the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=vali_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

model.eval()
with torch.no_grad():
    trainer.evaluate()
