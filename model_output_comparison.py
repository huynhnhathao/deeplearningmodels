import torch

import torch.nn as nn

from models.bert import (
    BertForClassification,
    BertForClassificationWithHFBertBase,
    BertForClassifierConfig,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_forward_pass():
    """Test the forward pass of the BERT model"""

    # Create test inputs
    # All tokens active

    # Initialize model

    # Run forward pass
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids, attention_mask)

    # Verify output shapes
    assert outputs.shape == (
        batch_size,
        config.num_classes,
    ), f"Expected output shape {(batch_size, config.num_classes)}, got {outputs.shape}"

    print("Forward pass test passed!")


def embedding_checkpoint(
    model: BertForClassification,
    modelhf: BertForClassificationWithHFBertBase,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    model_hidden_states = model.bert.embedding(input_ids)
    modelhf_hidden_states = modelhf.bert_base.embeddings(input_ids)
    assert model_hidden_states == modelhf_hidden_states

    # next pass to the first encoder layer
    for i in range(12):
        model_hidden_states = model.bert.encoder.encoder_layers[i](
            model_hidden_states, attention_mask
        )
        modelhf_hidden_states = modelhf.encoder.layer[i](
            modelhf_hidden_states, attention_mask
        )

        assert model_hidden_states == modelhf_hidden_states


def first_encoder_checkpoint(
    model: BertForClassification,
    modelhf: BertForClassificationWithHFBertBase,
    input_ids: torch.Tensor,
) -> None:
    pass


if __name__ == "__main__":
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(
        device
    )  # Random token IDs
    token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long).to(
        device
    )  # All type 0
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)
    checkpoint = "bert-base-uncased"
    config = BertForClassifierConfig(num_classes=2, device=device)
    model = BertForClassification(config)
    model.from_pretrained(checkpoint)
    modelhf = BertForClassificationWithHFBertBase(2, 768, checkpoint)
    model.to(device)
    modelhf.to(device)

    embedding_checkpoint(model, modelhf, input_ids, token_type_ids, attention_mask)
