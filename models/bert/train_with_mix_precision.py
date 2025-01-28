from typing import Callable, Optional, Tuple, List, Dict

from datasets import load_dataset

from transformers import DataCollatorWithPadding, BertTokenizer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LinearLR, LRScheduler
from torch.nn import CrossEntropyLoss


from tqdm.auto import tqdm

from models.bert import MyBertTokenizer, BertForClassification, BertForClassifierConfig


def val(
    model: nn.Module, dataloader: DataLoader, criterion: Callable, device: torch.device
) -> None:
    model.eval()
    model.to(device)

    num_examples = 0
    total_loss = 0
    correct_preds = 0

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
        total_loss += loss.item() * len(labels)

        predictions = torch.argmax(logits, -1)
        correct = (predictions == labels).sum().item()
        correct_preds += correct
    print(
        f"Avg loss: {total_loss/num_examples}, Accuracy: {correct_preds/num_examples}"
    )


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


def train(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable,
    train_dataloader: DataLoader,
    device: torch.device,
    grad_scaler: torch.amp.GradScaler,
    lr_scheduler: Optional[LRScheduler] = None,
) -> None:
    model.train()
    print(f"Initial grad scale: {grad_scaler.get_scale()}")
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids, token_type_ids, attention_mask)
            assert logits.dtype is torch.bfloat16
            loss = criterion(logits, labels)
            assert loss.dtype is torch.float32

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        print(f"Updated grad scale: {grad_scaler.get_scale()}")
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()


if __name__ == "__main__":
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # config = BertForClassifierConfig()
    # model = BertForClassification(config)
    # optimizer = AdamW(model.parameters(), lr=learning_rate)

    batch_size = 64
    train_dataloader, val_dataloader, test_dataloader = get_sst2_dataloaders(
        batch_size=batch_size
    )

    print("helloworld")
