import logging
import sys
from typing import Optional, Literal
import os
import shutil
from zipfile import ZipFile
from pathlib import Path
from huggingface_hub import hf_hub_download, upload_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def upload_file_to_hf(
    local_file_path: str,
    repo_id: str,
    repo_type: Literal["model", "dataset"],
    token: Optional[str] = None,
    path_in_repo: Optional[str] = None,
    commit_message: str = "Upload file",
) -> None:
    """
    Upload a file to Hugging Face hub.

    Args:
        local_file_path (str): Path to the local .pt checkpoint file
        repo_id (str): Repository ID in format "username/repo_name"
        repo_type (str, optional): Type of repository, either "model" or "dataset"
        token (str): Hugging Face authentication token. Read from environment variable HF_TOKEN if don't provide
        path_in_repo (str, optional): Destination path in the repository.
            Defaults to the filename from local_checkpoint_path
        commit_message (str, optional): Commit message for the upload

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        ValueError: If the repository ID is invalid
    """
    # Validate file exists
    if not os.path.isfile(local_file_path):
        raise FileNotFoundError(f"File not found: {local_file_path}")

    # Use filename as default path_in_repo if not specified
    if path_in_repo is None:
        path_in_repo = Path(local_file_path).name

    if token is None:
        logger.info("reading HF_TOKEN variable from environment")
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise RuntimeError("not found HF_TOKEN variable from environment")

    upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message=commit_message,
    )
    logger.info(f"Successfully uploaded {local_file_path} to {repo_id}/{path_in_repo}")
