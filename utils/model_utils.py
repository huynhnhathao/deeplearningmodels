import torch
from torch import nn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
