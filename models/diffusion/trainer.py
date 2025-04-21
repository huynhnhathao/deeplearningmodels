from typing import List, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from .model import UNet
from .scheduler import DiffusionScheduler


# the dataloader supposed to create noise


def get_train_val_dataloader(
    batch_size: int,
):
    train_dataset = torchvision.datasets.MNIST(
        root="./mnist", train=True, transform=torchvision.transforms.ToTensor
    )

    val_dataset = torchvision.datasets.MNIST(
        root="./mnist", train=False, transform=torchvision.transforms.ToTensor
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=diffusion_collate_fn,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        # collate_fn=diffusion_collate_fn,
    )

    return train_dataloader, val_dataloader


def set_null_class_with_probability(
    labels: torch.Tensor, null_class_idx: int, null_class_prob: float
) -> torch.Tensor:
    # randomly replace the true classes to the null class with a given probability
    rand = np.random.uniform(0, 1, labels.shape[0])
    selected_indices = rand < null_class_prob
    labels[selected_indices] = null_class_idx
    return labels


def train_diffuser_one_epoch(
    model: UNet,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.MSELoss,
    dataloader: DataLoader,
    max_timestep: int,
    scheduler: DiffusionScheduler,
    null_class_idx: int,
    null_class_prob: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        labels = set_null_class_with_probability(
            labels, null_class_idx, null_class_prob
        )

        # create a random set of timesteps from 1 to T inclusive
        t = np.random.randint(low=1, high=max_timestep + 1, size=images.shape[0])

        # shape (B, )
        alpha_bar_t = scheduler.get_alpha_bar(t, device=device)

        # noises are sampled from a standard normal distribution independently for each pixel of the image
        epsilon = torch.randn(size=images.shape, device=device)

        noisy_images = torch.sqrt(alpha_bar_t) * images + torch.sqrt(1 - alpha_bar_t)

        # labels shape (B, )
        # t shape (B, )
        logits = model(noisy_images, t, labels)
        optimizer.zero_grad()
        loss = criterion(logits, epsilon)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss
