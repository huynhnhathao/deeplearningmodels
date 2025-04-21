import os
from typing import List, Tuple, Optional
import logging
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataclasses import asdict

import numpy as np
from models.diffusion.model import UNet, DiffusionModelConfig
from models.diffusion.scheduler import DiffusionScheduler
from utils.huggingface import upload_file_to_hf


# the dataloader supposed to create noise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


@torch.no_grad
def eval(
    model: UNet,
    dataloader: DataLoader,
    criterion: torch.nn.MSELoss,
    scheduler: DiffusionScheduler,
    max_timestep: int,
    null_class_idx: int,
    null_class_prob: float,
    device: torch.device,
) -> float:
    """
    Run the evaluation of the model through the given dataloader and return the average loss
    """
    model.eval()

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
        loss = criterion(logits, epsilon)
        total_loss += loss.detach().cpu().item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss


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


def train_diffuser(
    model_config: DiffusionModelConfig,
    num_epoch: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    upload_model: Optional[int],
    hf_repo_id: str,
) -> UNet:

    model = UNet(model_config)

    model = model.to(device)

    diffusion_linear_scheduler = DiffusionScheduler()

    train_dataloader, val_dataloader = get_train_val_dataloader(batch_size)

    optimizer = Adam(params=model.parameters(), lr=lr)

    criterion = torch.nn.MSELoss()

    for epoch in range(num_epoch):
        train_loss = train_diffuser_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_dataloader,
            max_timestep=model_config.max_scheduler_steps,
            scheduler=diffusion_linear_scheduler,
            null_class_idx=10,
            null_class_prob=0.2,
            device=device,
        )

        val_loss = eval(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            scheduler=diffusion_linear_scheduler,
            max_timestep=model_config.max_scheduler_steps,
            null_class_idx=10,
            null_class_prob=0.2,
            device=device,
        )

        logger.info(
            f"Epoch {epoch+1} Train loss {train_loss:.2f}, Val loss: {val_loss:.2f}"
        )

        if upload_model is not None and (epoch + 1) % upload_model == 0:
            data = {
                "config": model_config,
                "scheduler": asdict(diffusion_linear_scheduler),
                "statedict": model.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            save_path = f"./models/diffusion_epoch{epoch+1}.pt"
            torch.save(data, save_path)

            upload_file_to_hf(
                local_file_path=save_path,
                repo_id=hf_repo_id,
                repo_type="model",
            )
