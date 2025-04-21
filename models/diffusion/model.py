import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class DiffusionModelConfig:
    image_height: int
    image_width: int
    image_chan: int

    embedding_dim: int
    max_scheduler_steps: int
    # number of classes in the training image
    # add one class for the "no class" class, this class mean it can be any of the other classes
    num_class: int


class UNet(nn.Module):
    def __init__(self, config: DiffusionModelConfig) -> None:
        """
        height, width: height and width of the input image
        """
        # timestep embedding will be broadcasted and added to the noised input image
        self.time_embedding = nn.Embedding(
            num_embeddings=config.max_scheduler_steps,
            embedding_dim=config.embedding_dim,
        )

        # the class information will be broadcasted and added to the input image too
        self.class_embedding = nn.Embedding(
            num_embeddings=config.num_class,
            embedding_dim=config.embedding_dim,
        )

        self.encoder = nn.ModuleDict(
            {
                "conv1": nn.Conv2d(
                    in_channels=config.image_chan,
                    out_channels=64,
                    kernel_size=3,
                    padding="same",
                    stride=1,
                ),  # (64, 28, 28)
                "conv2": nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    padding="same",
                    stride=1,
                ),  # (64, 28, 28)
                "maxpool1": nn.MaxPool2d(kernel_size=3, stride=1),  # (64, 14, 14)
                "conv3": nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    padding="same",
                    stride=1,
                ),  # (128, 14, 14)
                "conv4": nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    padding="same",
                    stride=1,
                ),  # (128, 14, 14)
                "maxpool2": nn.MaxPool2d(kernel_size=3, stride=1),  # (128, 7, 7)
                "conv5": nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    padding="same",
                    stride=1,
                ),  # (256, 7, 7)
                "conv6": nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding="same",
                    stride=1,
                ),  # (256, 7, 7)
            }
        )

        self.decoder = nn.ModuleDict(
            {
                "transpose_conv1": nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=3,
                    stride=2,
                ),  # (128, 14, 14)? one skip conn of the encoder.conv4
                "conv1": nn.Conv2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding="valid",
                ),  # (256, 14, 14)
                "conv2": nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),  # (256, 14, 14)
                "transpos_conv2": nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),  # (128, 28, 28) + one skip conn of the encoder.conv2
                "conv3": nn.Conv2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),  # (128, 28, 28)
                "conv4": nn.Conv2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),  # (64, 28, 28) + one skip conn at the encoder.conv1
                "conv5": nn.Conv2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),  # (64, 28, 28)
                "conv6": nn.Conv2d(
                    in_channels=64,
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),  # (1, 28, 28)
            }
        )

    def forward(self, x: torch.Tensor, t: int, c: int) -> torch.Tensor:
        # x shape (B, C, H, W)
        t_embedding = self.time_embedding(t).expand(1, 1, 1, -1)
        c_embedding = self.class_embedding(c).expand(1, 1, 1, -1)

        # just add everything together :)
        x = x + t_embedding + c_embedding
