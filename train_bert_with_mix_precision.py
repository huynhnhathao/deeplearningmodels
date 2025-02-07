from typing import Callable, Optional, Tuple, List, Dict

import os

from datasets import load_dataset

from transformers import DataCollatorWithPadding, BertTokenizer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LinearLR, LRScheduler
from torch.nn import CrossEntropyLoss

import mlflow

from tqdm.auto import tqdm
import argparse

from sklearn.metrics import f1_score

from models.bert import (
    BertForClassification,
    BertForClassifierConfig,
)

mlflow.set_experiment("finetune-bert-on-sst2")


def val(
    model: nn.Module, dataloader: DataLoader, criterion: Callable, device: torch.device
) -> Tuple[float, float]:
    total_loss = 0
    correct_preds = 0
    num_examples = 0

    for batch in dataloader:
        input_ids, token_type_ids, attention_mask = (
            batch["input_ids"].to(device),
            batch["token_type_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        labels = batch["labels"].to(device)
        num_examples += len(labels)

        logits = model(input_ids, token_type_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        predictions = torch.argmax(logits, -1)
        correct = (predictions == labels).sum().item()
        correct_preds += correct

    avg_val_loss = total_loss / len(dataloader)
    val_acc = correct_preds / num_examples
    print(f"Validation loss: {avg_val_loss}, Validation accuracy: {val_acc}")
    return (avg_val_loss, val_acc)


def get_sst2_dataloaders(
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    checkpoint = "bert-base-uncased"
    raw_datasets = load_dataset("glue", "sst2")
    tokenizer = BertTokenizer.from_pretrained(checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    raw_datasets = raw_datasets.map(tokenize_function, batched=True)
    raw_datasets = raw_datasets.remove_columns(["sentence", "idx"])
    raw_datasets = raw_datasets.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer, padding=True)

    train_dataloader = DataLoader(
        dataset=raw_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_dataloader = DataLoader(
        dataset=raw_datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    test_dataloader = DataLoader(
        dataset=raw_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    return (train_dataloader, val_dataloader, test_dataloader)


def log_training_config(training_config: argparse.Namespace):
    mlflow.log_params(
        {
            "cls_dropout_prob": training_config.cls_dropout_prob,
            "l2_regularization": training_config.l2_regularization,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "num_epoch": training_config.num_epoch,
        }
    )


def log_model_artifact(model: nn.Module, artifact_path: str):
    mlflow.log_artifact("bert_sst2_model_summary.txt", artifact_path)


# train the model one epoch
def train(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable,
    train_dataloader: DataLoader,
    device: torch.device,
    grad_scaler: torch.amp.GradScaler,
    lr_scheduler: Optional[LRScheduler] = None,
    progress_bar=None,
) -> Tuple[float, float]:
    total_loss = 0
    total_examples = 0
    correct_preds = 0

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        total_examples += len(labels)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids, token_type_ids, attention_mask)
            assert logits.dtype is torch.bfloat16
            loss = criterion(logits, labels)
            assert loss.dtype is torch.float32

        total_loss += loss.item()
        predictions = torch.argmax(logits, -1)
        correct = (predictions == labels).sum().item()
        correct_preds += correct

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()
        if progress_bar is not None:
            progress_bar.update(1)

    avg_training_loss = total_loss / len(train_dataloader)
    accuracy = correct_preds / total_examples
    print(f"Training loss {avg_training_loss}, Training Accuracy: {accuracy}")
    return (avg_training_loss, accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="learning rate for AdamW"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=3, help="number of epoch to train"
    )

    parser.add_argument(
        "--cls_dropout_prob",
        type=float,
        default=0.3,
        help="dropout prob apply to the classification fcnn intermediate layer",
    )
    parser.add_argument(
        "--l2_regularization", type=float, default=0.001, help="optimizer weight decay"
    )

    parser.add_argument(
        "--artifact_path",
        type=str,
        default="",
        help="path to folder to store model artifacts",
    )
    parser.add_argument(
        "--state_dict_path",
        type=str,
        default="",
        help="path to the Pytorch saved state dict to continue training",
    )

    training_config = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = BertForClassifierConfig(2, device, training_config.cls_dropout_prob)
    model = BertForClassification(config)
    criterion = CrossEntropyLoss()
    grad_scaler = torch.amp.GradScaler(device.type)
    optimizer = Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.l2_regularization,
    )
    train_dataloader, val_dataloader, test_dataloader = get_sst2_dataloaders(
        batch_size=training_config.batch_size
    )
    num_steps = training_config.num_epoch * len(train_dataloader)
    lr_scheduler = LinearLR(
        optimizer=optimizer, start_factor=1, end_factor=0.05, total_iters=num_steps
    )
    progress_bar = tqdm(range(training_config.num_epoch * len(train_dataloader)))

    if training_config.state_dict_path != "":
        print(f"loading state dict from {training_config.state_dict_path}")
        model.load_state_dict(
            torch.load(training_config.state_dict_path, weights_only=True), strict=True
        )
    else:
        checkpoint = "bert-base-uncased"
        print(f"loading pretrained weights from huggingface checkpoint {checkpoint}")
        model.from_pretrained(checkpoint)

    with mlflow.start_run():
        log_training_config(training_config)

        # log_model_artifact(model, training_config.artifact_path)

        model.to(device)
        print("Start training")
        for epoch in range(training_config.num_epoch):
            model.train()
            avg_training_loss, training_acc = train(
                model,
                optimizer,
                criterion,
                train_dataloader,
                device,
                grad_scaler,
                lr_scheduler,
                progress_bar,
            )

            model.eval()
            with torch.no_grad():
                avg_val_loss, val_acc = val(model, val_dataloader, criterion, device)
            state_dict_file_path = os.path.join(
                training_config.artifact_path, f"bert_epoch{epoch}.pt"
            )
            torch.save(model.state_dict(), state_dict_file_path)

            mlflow.log_metrics(
                {
                    "training_loss": avg_training_loss,
                    "training_acc": training_acc,
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
