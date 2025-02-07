from typing import Callable, Union, Tuple


import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from sklearn.metrics import f1_score, accuracy_score


def val(
    model: nn.Module, dataloader: DataLoader, criterion: Callable, device: torch.device
) -> Tuple[float, float, float]:
    total_loss = 0
    num_examples = 0
    y_pred = []
    y_true = []
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

        y_pred.extend(torch.argmax(logits, -1).tolist())
        y_true.extend(labels.tolist())

    avg_val_loss = total_loss / len(dataloader)

    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred)
    print(
        f"Validation loss: {avg_val_loss}, Validation accuracy: {val_acc}, f1 score: {val_f1}"
    )
    return (avg_val_loss, val_acc, val_f1)


# train the model one epoch
def train(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: Callable,
    train_dataloader: DataLoader,
    device: torch.device,
    grad_scaler: torch.amp.GradScaler,
    lr_scheduler: Union[LRScheduler, None] = None,
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
