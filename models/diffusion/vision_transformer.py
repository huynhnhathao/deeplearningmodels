import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from dataclasses import dataclass
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class VTConfig:
    image_height: int
    image_width: int
    channel: int
    # one number, represent a square patch size (n, n),
    # it should be divisible by both the image_width and image_height
    patch_size: tuple[int, int]
    # transformers config
    hidden_size: int
    num_head: int
    transformer_encoder_num_layers: int


@dataclass
class VTForClassifierConfig(VTConfig):
    num_classes: int


class VisionTransformer(nn.Module):
    """
    VisionTransformer model take an input image of shape (H, W, C) and transform them into n embedding vectors
    of shape (n+1, h) where n is the number of patch, h is the hidden dimension

    This model also prepend a CLS token to the start of the patch, and use that CLS token embedding to do classification
    This forces the model to aggregate the image's representation into the embedding of the CLS token, what kind of representation
    depends on your training objective though, this method is a way to do aggregation of the image into one embedding vector
    """

    def __init__(self, config: VTConfig):
        super().__init__()
        self.config = config

        assert len(config.patch_size) == 2, "requires a 2D patch_size"
        assert (
            config.image_height % config.patch_size[0] == 0
        ), "image_height is not divisible by patch_size[0]"
        assert (
            config.image_width % config.patch_size[1] == 0
        ), "image_width is not divisible by patch_size[1]"

        # learn positional embedding
        flatten_patch_size = (
            config.patch_size[0] * config.patch_size[1] * config.channel
        )
        self.linear = nn.Linear(flatten_patch_size, config.hidden_size)
        # how many equal patches the input image can be split into
        self.num_patches = (config.image_height * config.image_width) // (
            config.patch_size[0] * config.patch_size[1]
        )
        self.positional_embedding = nn.Embedding(
            self.num_patches + 1, config.hidden_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=config.num_head, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, config.transformer_encoder_num_layers
        )

        self.CLS_TOKEN_EMBEDDING = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expecting x of shape (B, H, W, C), channel last
        """
        B, H, W, C = x.shape
        assert (
            H == self.config.image_height
        ), f"x shape: {x.shape}, expecting {H} == {self.config.image_height}"
        assert (
            W == self.config.image_width
        ), f"x shape: {x.shape}, expecting {W} == {self.config.image_width}"
        assert (
            C == self.config.channel
        ), f"x shape: {x.shape}, expecting channel dim of size {self.config.channel}"

        patches = x.view(
            B,
            H // self.config.patch_size[0],
            self.config.patch_size[0],
            W // self.config.patch_size[1],
            self.config.patch_size[1],
            C,
        )
        patches = patches.permute(0, 1, 3, 2, 4, -1).contiguous()
        # (B, num_patches, flatten_patch_dim)
        patches = patches.view(
            B, -1, self.config.patch_size[0] * self.config.patch_size[1] * C
        )

        # first projection of the flatten patch (h, w, c) to the transformer's hidden size
        h = self.linear(patches)

        # prepend the CLS token
        h = torch.cat([h, self.CLS_TOKEN_EMBEDDING.expand(B, 1, -1)], dim=1)

        # add positional embedding vectors
        patch_indices = torch.arange(start=0, end=self.num_patches + 1).expand(1, -1)
        position_embedding = self.positional_embedding(patch_indices)
        h = h + position_embedding

        # forward through the transformer
        # expecting shape (B, num_patches, hidden_size)
        out: torch.Tensor = self.transformer_encoder(h)
        assert out.shape[0] == B
        assert (
            out.shape[1] == self.num_patches + 1
        ), f"expecting second dimension of the output tensor equals {self.num_patches}, received {out.shape}"
        return out


class VisionTransformerForClassifier(nn.Module):
    def __init__(self, config: VTForClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_transformer = VisionTransformer(config)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expecting x of shape (B, H, W, C), channel last
        """
        B, H, W, C = x.shape
        # h shape (B, num_patches, hidden_dim)
        h = self.vision_transformer(x)
        # take the CLS token embedding
        first_h = h[:, 0, :]

        logits: torch.Tensor = self.classifier(first_h)
        assert logits.shape[0] == B and logits.shape[1] == self.config.num_classes
        return logits


def get_cifar10_dataloader(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transforms = torchvision.transforms.Compose([ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(
        root="./", train=True, transform=transforms, download=True
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root="./", train=False, transform=transforms, download=True
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, val_dataloader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    return np.mean(losses)


def val(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()
    losses = []
    total_correct: int = 0
    total: int = 0

    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        logits: torch.Tensor = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.detach().cpu().item())
        preds = torch.argmax(logits, -1)
        total_correct += torch.sum(preds == labels)
        total += len(batch)

    return np.mean(losses), total_correct / total


def calculate_model_memory(model: nn.Module):
    """
    Calculate and print the memory usage of a PyTorch model's parameters based on their detected data type.

    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    # Dictionary mapping PyTorch dtypes to bytes per parameter
    bytes_per_param_dict = {
        torch.float32: 4,  # 32 bits = 4 bytes
        torch.float16: 2,  # 16 bits = 2 bytes
        torch.int8: 1,  # 8 bits = 1 byte
        torch.int32: 4,  # 32 bits = 4 bytes
        torch.int64: 8,  # 64 bits = 8 bytes
    }

    # Detect the data type from the first parameter
    param_iter = iter(model.parameters())
    try:
        first_param = next(param_iter)
        dtype = first_param.dtype
    except StopIteration:
        print("Model has no parameters!")
        return

    # Get bytes per parameter based on detected dtype
    # Default to 4 bytes if dtype not found
    bytes_per_param = bytes_per_param_dict.get(dtype, 4)
    dtype_name = str(dtype).replace("torch.", "")  # Clean up dtype name for printing

    # Count total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Count total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate total memory in bytes
    total_memory_bytes = total_params * bytes_per_param

    # Convert to KB, MB, and GB for readability
    total_memory_kb = total_memory_bytes / 1024
    total_memory_mb = total_memory_kb / 1024
    total_memory_gb = total_memory_mb / 1024

    # Print results
    logger.info(f"Model Memory Usage (Detected dtype: {dtype_name}):")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Total Memory: {total_memory_gb:,.2f} GB")


if __name__ == "__main__":
    config = VTForClassifierConfig(
        image_height=32,
        image_width=32,
        channel=3,
        patch_size=(8, 8),
        hidden_size=512,
        num_head=8,
        transformer_encoder_num_layers=3,
        num_classes=10,
    )
    model = VisionTransformerForClassifier(config)

    calculate_model_memory(model)

    B = 2
    input = torch.randn(B, 32, 32, 3)

    out: torch.Tensor = model(input)

    assert out.shape[0] == B
    assert out.shape[1] == config.num_classes
