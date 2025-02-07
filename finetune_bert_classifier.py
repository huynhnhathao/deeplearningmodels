from typing import Callable, Optional, Tuple, List, Dict

import os

from datasets import load_dataset

from transformers import DataCollatorWithPadding, BertTokenizer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.nn import CrossEntropyLoss

import mlflow

from tqdm.auto import tqdm
import argparse


from models.bert import BertForClassificationWithHFBertBase

from models.bert.trainer import train, val

mlflow.set_experiment("finetune-bert-on-sst2-with-hf-implementation")


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
        "--l2_regularization", type=float, default=0, help="optimizer weight decay"
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
    model = BertForClassificationWithHFBertBase(2, 768, "bert-base-uncased")

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
        optimizer=optimizer, start_factor=1, end_factor=0.1, total_iters=num_steps
    )
    progress_bar = tqdm(range(training_config.num_epoch * len(train_dataloader)))

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
                avg_val_loss, val_acc, f1_score = val(
                    model, val_dataloader, criterion, device
                )
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
                    "val_f1": f1_score,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
