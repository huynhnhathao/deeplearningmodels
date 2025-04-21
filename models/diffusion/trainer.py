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


def train_diffuser_one_epoch(
    model: UNet,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.MSELoss,
    dataloader: DataLoader,
    max_timestep: int,
    scheduler: DiffusionScheduler,
    null_class_prob: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)

        # noises are sampled from a standard normal distribution independently for each pixel of the image
        epsilon = torch.randn(size=images.shape, device=device)

        # create a random set of timesteps from 1 to T inclusive
        t = np.random.randint(low=1, high=max_timestep + 1, size=images.shape[0])

        # shape (B, )
        alpha_bar_t = scheduler.get_alpha_bar(t, device=device)
        alpha_bar_t = alpha_bar_t.expand(1, 1, images.shape[-2:])

        noisy_images = (
            torch.sqrt(alpha_bar_t) * images + torch.sqrt(1 - alpha_bar_t) * epsilon
        )

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
