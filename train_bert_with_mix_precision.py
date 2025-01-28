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
import argparse


from models.bert import (
    BertForClassification,
    BertForClassifierConfig,
)


def val(
    model: nn.Module, dataloader: DataLoader, criterion: Callable, device: torch.device
) -> None:
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
    print(
        f"Avg loss: {total_loss/len(dataloader)}, Accuracy: {correct_preds/num_examples}"
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
) -> None:
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
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()
        if progress_bar is not None:
            progress_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="learning rate for AdamW"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=5, help="number of epoch to train"
    )
    parser.add_argument(
        "--state_dict_path",
        type=str,
        default="",
        help="path to the Pytorch saved state dict to continue training",
    )

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device)
    config = BertForClassifierConfig(2, device)
    model = BertForClassification(config)
    if args.state_dict_path != "":
        model.load_state_dict(
            torch.load(args.state_dict_path, weights_only=True), strict=True
        )
    else:
        checkpoint = "bert-base-uncased"
        model.from_pretrained(checkpoint)

    criterion = CrossEntropyLoss()
    grad_scaler = torch.amp.GradScaler(device.type)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    train_dataloader, val_dataloader, test_dataloader = get_sst2_dataloaders(
        batch_size=args.batch_size
    )
    num_steps = args.batch_size * len(train_dataloader)
    lr_scheduler = LinearLR(
        optimizer=optimizer, start_factor=1, end_factor=0.1, total_iters=num_steps
    )

    model.to(device)
    progress_bar = tqdm(range(args.num_epoch * len(train_dataloader)))
    for epoch in range(args.num_epoch):
        model.train()
        train(
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
        val(model, val_dataloader, criterion, device)
        torch.save(model.state_dict(), f"./bert_epoch{epoch}.pt")
