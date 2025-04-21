import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class DiffusionModelConfig:
    image_height: int = 28
    image_width: int = 28
    image_chan: int = 1

    embedding_dim: int = 28 * 28

    max_scheduler_steps: int = 1000
    # number of classes in the training image
    # add one class for the "no class" class, this class mean it can be any of the other classes
    num_class: int = 11  # plus the unconditional class


class UNet(nn.Module):
    def __init__(self, config: DiffusionModelConfig) -> None:
        """
        height, width: height and width of the input image
        """
        super().__init__()
        self.config = config
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
                    out_channels=128,
                    kernel_size=3,
                    padding="same",
                    stride=1,
                ),  # (128, 28, 28)
                "maxpool1": nn.MaxPool2d(kernel_size=2),  # (128, 14, 14)
                "conv3": nn.Conv2d(
                    in_channels=128,
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
                "maxpool2": nn.MaxPool2d(kernel_size=2),  # (128, 7, 7)
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
                    kernel_size=2,
                    stride=2,
                ),
                # (128, 14, 14) one skip conn of the encoder.conv4
                "conv1": nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),  # (256, 14, 14)
                "conv2": nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding="same",
                ),  # (256, 14, 14)
                "transpose_conv2": nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=2,
                    stride=2,
                ),
                # (128, 28, 28) + one skip conn at the encoder.conv2
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
                ),
                # (64, 28, 28) + one skip conn at the encoder.conv1
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

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """

        Args
            x: tensor shape (B, C, H, W),  channel first
        """
        # t_embedding and c_embedding are of shape (1, embedding_dim)
        t_embedding = (
            self.time_embedding(t)
            .view(self.config.image_height, self.config.image_width)
            .expand(1, 1, self.config.image_height, self.config.image_width)
        )
        c_embedding = (
            self.class_embedding(c)
            .view(self.config.image_height, self.config.image_width)
            .expand(1, 1, self.config.image_height, self.config.image_width)
        )

        tc = t_embedding + c_embedding

        # just add everything together :)
        h = x + tc
        hidden_states = []
        for layer_name, layer in self.encoder.items():
            h = layer(h)
            if layer_name in ["conv1", "conv2", "conv4"]:
                hidden_states.append(h)

        for layer_name, layer in self.decoder.items():
            if layer_name in ["conv1", "conv3", "conv5"]:
                # dim1 is the channel dim
                h = torch.concat([hidden_states.pop(), h], dim=1)
            h = layer(h)
        # h now is of shape (1, 28, 28)
        return h


if __name__ == "__main__":
    config = DiffusionModelConfig(
        image_height=28,
        image_width=28,
    )
    model = UNet(config)
    print(model)
    input = torch.randn((8, 1, 28, 28))
    t = torch.tensor([100])
    c = torch.tensor([10])
    out = model(input, t, c)
    # out shape  (8, 1, 28, 28)
