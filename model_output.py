import torch

import torch.nn as nn

from models.bert import (
    BertForClassification,
    BertForClassificationWithHFBertBase,
    BertForClassifierConfig,
)

torch.manual_seed(0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def embedding_checkpoint(
    model: BertForClassification,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(0, seq_length, device=device).expand(batch_size, -1)

    model_hidden_states = model.bert.embedding(input_ids, token_type_ids, position_ids)

    # next pass to the first encoder layer
    for i in range(12):
        model_hidden_states = model.bert.encoder.encoder_layers[
            i
        ].multi_head_self_attention(model_hidden_states, attention_mask)

        print(model_hidden_states.shape)


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
    model.to(device)

    model.eval()
    embedding_checkpoint(model, input_ids, token_type_ids, attention_mask)
