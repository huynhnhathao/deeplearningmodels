import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class VTConfig:
    image_height: int
    image_width: int
    # one number, represent a square patch size (n, n), 
    # it should be divisible by both the image_width and image_height
    patch_size: tuple[int, int]
    hidden_size: int
    transformer_encoder_num_layers: int
    num_head: int
    num_patches: int # (image_height * image_width) // patch_size[0] * patch_size[1]

@dataclass
class VTForClassifierConfig(VTConfig):
    num_classes: int


class VisionTransformer(nn.Module):
    """
    VisionTransformer model take an input image of shape (H, W, C) and transform them into n embedding vectors
    of shape (n, h) where n is the number of patch, h is the hidden dimension
    """
    def __init__(self, config: VTConfig):
        super().__init__()
        self.config = config

        assert len(config.patch_size) == 2, "requires a 2D patch_size"
        assert config.image_height % config.patch_size[0] == 0, "input image is not divisible by patch_size[0]"
        assert config.image_width % config.patch_size[1] == 0, "input image is not divisible by patch_size[1]"

        # learn positional embedding 
        flatten_patch_size = config.patch_size[0] * config.patch_size
        self.linear = nn.Linear(flatten_patch_size, config.hidden_size) 
        num_patch = (config.image_height * config.image_width) // (config.patch_size[0] * config.patch_size[1])
        self.positional_embedding = nn.Embedding(num_patch, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.transformer_encoder_num_layers )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert H == self.config.image_height
        assert W == self.config.image_width
        patches = x.view(B, H//self.config.patch_size[0], self.config.patch_size[0], W // self.config.patch_size[1], self.config.patch_size[1], C)
        patches = patches.permute(0, 1, 3, 2, 4, -1)
        patches = patches.view(B, -1, self.config.patch_size[0] * self.config.patch_size[1] * C)
        
        # first projection of the flatten patch (h, w, c) to the transformer's hidden size
        h = self.linear(patches)

        # add positional embedding vectors
        patch_indices = torch.arange(start=0, end=self.config.num_patches).expand(1, -1)
        position_embedding = self.positional_embedding(patch_indices)
        h = h + position_embedding

        # forward through the transformer
        # expecting shape (B, num_patches, hidden_size)
        out = self.transformer_encoder(h)
        
        return out


class VisionTransformerForClassifier(nn.Module):
    def __init__(self, config: VTForClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_transformer = VisionTransformer(config)
        self.classifier = nn.Linear(self.config.num_patches * self.config.hidden_size, self.config.num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return tensor logits of shape (B, num_classes)
        """
        B, H, W, C = x.shape

        h = self.vision_transformer(x)
        logits: torch.Tensor = self.classifier(h)
        assert logits.shape[0] == B and logits.shape[1] == self.config.num_classes
        return logits






