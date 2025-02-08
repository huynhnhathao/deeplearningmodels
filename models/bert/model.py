from dataclasses import dataclass
import logging
import math

import torch
from torch.nn import functional as F
import torch.nn as nn


from transformers import (
    BertModel as TransformerBertModel,
    BertTokenizer,
    DataCollatorWithPadding,
)
from typing import Callable

logger = logging.getLogger()

VERY_NEGATIVE_NUMBER = -1e20


@dataclass
class BertConfig:
    hidden_dim: int = 768
    max_sequence_length: int = 512
    vocab_size: int = 30522
    type_vocab_size: int = 2
    padding_token_id: int = 0

    # self-attention parameters
    num_heads: int = 12
    fcnn_middle_dim: int = 3072
    # all dropout layers used in the model has this same dropout probability
    dropout_prob: float = 0.1

    num_encoder_layers: int = 12
    layer_norm_eps: float = 1e-12


@dataclass
class BertForClassifierConfig(BertConfig):
    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.device = device


class BertEmbedding(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.hidden_dim
        )
        self.token_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_dim
        )
        self.words_embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.padding_token_id
        )
        # the original transformers encoder applies dropout to the sum of embeddings and positional embeddings
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_dim, config.layer_norm_eps)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        token_ids, token_type_ids, position_ids shape: (batch_size, sequence_length)
        token_ids: vocab indices of the tokens
        token_type_ids: either 0 or 1 per position in the sequence, use to differentiate two sentences
        position_ids: count from 0 at the start of the sequence to max_sequence_length
        """
        embeddings = (
            self.words_embedding(token_ids)
            + self.token_type_embedding(token_type_ids)
            + self.position_embedding(position_ids)
        )

        return self.dropout(self.layer_norm(embeddings))


class EncoderLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention(
            config.hidden_dim,
            config.num_heads,
            config.dropout_prob,
            config.layer_norm_eps
        )

        self.ffnn = FFNN(config.hidden_dim, config.fcnn_middle_dim, config.dropout_prob, config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_output = self.multi_head_self_attention(hidden_states.clone(), attention_mask)
        hidden_states =  self.ffnn(attention_output)
        return hidden_states


class FFNN(nn.Module):
    def __init__(self, hidden_dim: int, middle_dim: int, dropout_prob: float, layer_norm_eps: float) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(hidden_dim, middle_dim, bias=True)
        self.last_dense = nn.Linear(middle_dim, hidden_dim, bias=True)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.intermediate_dense(inputs)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.last_dense(hidden_states)
        return self.layer_norm(self.dropout(hidden_states) + inputs)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout_prob: float,
        layer_norm_eps: float
    ):
        """
        Transformer Self-attention
            Args
                - hidden_dim: size of the token embedding and hidden_state
                - num_heads: number of attention head
                - dropout_prob: dropout probability apply to the attention scores,
                    practically mean randomly prevent the attention to attend to some random tokens
                - linear_dropout_prob: dropout prob apply to the linear layer after the self-attention layer
        """
        super().__init__()
        assert (
            hidden_dim % num_heads == 0
        ), "hidden_dim must divisible for num_heads in self-attention"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # we'll use dk = dq = dv
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout_prob = dropout_prob
        # because hidden_dim % num_heads == 0,
        # output dim for num_heads is also the embedding_dim
        self.Q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.K = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.V = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.attention_dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_dim, layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states shape: (batch_size, sequence_length, hidden_dim)
        """
        batch_size, sequence_length, hidden_dim = hidden_states.size()
        # (batch_size, sequence_length, hidden_dim)
        queries: torch.Tensor = self.Q(hidden_states)
        keys: torch.Tensor = self.K(hidden_states)
        values: torch.Tensor = self.V(hidden_states)

        # transpose the queries, keys and values to separate the head dimension out
        queries = queries.view(
            batch_size, self.num_heads, sequence_length, self.head_dim
        )
        keys = keys.view(batch_size, self.num_heads, sequence_length, self.head_dim)
        values = values.view(batch_size, self.num_heads, sequence_length, self.head_dim)

        
        # test the torch's implementation
        attention_values = torch.nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )

        # #-----------------------------------------------------------
        # # attention_scores shape (batch_size, num_heads, sequence_length, sequence_length)
        # # swap keys to have shape (batch_size, self.num_heads, self.head_dim, sequence_length)
        # attention_scores = torch.matmul(queries, torch.transpose(keys, -1, -2))
        # attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim))
        # # attention mask is of shape (batch_size, sequence_length),
        # # need to create a tensor of shape (batch_size, num_heads, sequence_length, sequence_length)
        # # to fit the attention_scores tensor
        # # the element-wise logical end operation is to create an attention square matrix such that
        # # padded tokens can't be attended to and can not attend to other tokens
        # attention_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(-1)
        # attention_scores = attention_scores.masked_fill(
        #     attention_mask.unsqueeze(1) == 0, VERY_NEGATIVE_NUMBER
        # )
        # attention_scores = F.softmax(attention_scores, -1)

        # # apply dropout to the attention scores, random tokens will not be attended to at training
        # attention_scores = self.attention_dropout(attention_scores)

        # # attention_values shape: (batch_size, num_heads, sequence_length, dk)
        # attention_values = torch.matmul(attention_scores, values)

        # # concat all heads' values to one vector representation per token
        # # (batch_size, sequence_length, hidden_dim)
        # # this is the "concat then linear projection step" in the Attention is all you need paper

        # # -----------------------------------------------------------
        attention_values = attention_values.view(batch_size, sequence_length, -1)
        attention_values = self.dense(attention_values)
        return self.layer_norm(self.dropout(attention_values) + hidden_states)


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        """
        Extract the first token embedding vector out of the hidden_states tensor
        then apply a linear and a tanh function to it
        """
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # expecting the hidden_states shape of (batch_size, sequence_length, hidden_dim)
        cls_embedding = hidden_states[:, 0, :]
        return self.tanh(self.linear(cls_embedding))


class BertModel(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.embedding = BertEmbedding(config)
        self.encoder = BertEncoder(config)
        self.max_sequence_length = config.max_sequence_length

    def forward(
        self,
        token_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """
        Input tensors to this module is expected to be already padded to the model's max_sequence_length
        """
        self.validate_input_dimension(
            token_ids, token_type_ids, position_ids, attention_mask
        )

        # input_ids shape: (batch_size, sequence_length)
        # sequence_length must match the max_sequence_length, this module expects all padding be done before it
        embeddings = self.embedding(token_ids, token_type_ids, position_ids)
        # hidden_states shape (batch_size, sequence_length, hidden_dim)
        hidden_states: torch.Tensor = self.encoder(embeddings, attention_mask)
        return (hidden_states, )

    def validate_input_dimension(
        self,
        token_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        # check that all the dimensions match
        assert (
            token_ids.shape[0]
            == token_type_ids.shape[0]
            == position_ids.shape[0]
            == attention_mask.shape[0]
        ), "input batch dimensions don't match"

        assert (
            token_ids.shape[1]
            == token_type_ids.shape[1]
            == position_ids.shape[1]
            == attention_mask.shape[1]
        ), "input sequence_length dimensions don't match"

        assert torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ).item(), "attention_mask must only contains 0s or 1s"


class BertForClassification(nn.Module):
    def __init__(self, config: BertForClassifierConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.classification_head = nn.Linear(
            config.hidden_dim, config.num_classes, bias=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.size()
        if sequence_length > self.config.max_sequence_length:
            logger.warning(
                f"input sequences' length {sequence_length} exceeded the model's max_sequence_length of {self.config.max_sequence_length}, they will be truncated"
            )
            input_ids = input_ids[:, : self.config.max_sequence_length]
        position_ids = torch.arange(0, sequence_length, device=self.config.device)

        hidden_states = self.bert(
            input_ids,
            token_type_ids,
            position_ids.expand(batch_size, -1),
            attention_mask,
        )
        return self.classification_head(hidden_states[0][:, 0, :])
    
    def from_pretrained(self, model_name_or_path: str) -> nn.Module:
        """
        Load weights of a model identifier from huggingface's model hub
        """
        print(f"loading pretrained weights from {model_name_or_path}")
        pretrained_model = TransformerBertModel.from_pretrained(
            model_name_or_path, torch_dtype="auto"
        )
        state_dict_mapping = {
            # Embedding layers
            "embeddings.word_embeddings.weight": "bert.embedding.words_embedding.weight",
            "embeddings.position_embeddings.weight": "bert.embedding.position_embedding.weight",
            "embeddings.token_type_embeddings.weight": "bert.embedding.token_type_embedding.weight",
            "embeddings.LayerNorm.weight": "bert.embedding.layer_norm.weight",
            "embeddings.LayerNorm.bias": "bert.embedding.layer_norm.bias",
            # Encoder layers
            **{
                f"encoder.layer.{i}.attention.self.query.weight": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.Q.weight"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.attention.self.query.bias": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.Q.bias"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.attention.self.key.weight": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.K.weight"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.attention.self.key.bias": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.K.bias"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.attention.self.value.weight": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.V.weight"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.attention.self.value.bias": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.V.bias"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.attention.output.dense.weight": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.dense.weight"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.attention.output.dense.bias": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.dense.bias"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.attention.output.LayerNorm.weight": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.layer_norm.weight"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.attention.output.LayerNorm.bias": f"bert.encoder.encoder_layers.{i}.multi_head_self_attention.layer_norm.bias"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.intermediate.dense.weight": f"bert.encoder.encoder_layers.{i}.ffnn.intermediate_dense.weight"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.intermediate.dense.bias": f"bert.encoder.encoder_layers.{i}.ffnn.intermediate_dense.bias"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.output.dense.weight": f"bert.encoder.encoder_layers.{i}.ffnn.last_dense.weight"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.output.dense.bias": f"bert.encoder.encoder_layers.{i}.ffnn.last_dense.bias"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.output.LayerNorm.weight": f"bert.encoder.encoder_layers.{i}.ffnn.layer_norm.weight"
                for i in range(self.config.num_encoder_layers)
            },
            **{
                f"encoder.layer.{i}.output.LayerNorm.bias": f"bert.encoder.encoder_layers.{i}.ffnn.layer_norm.bias"
                for i in range(self.config.num_encoder_layers)
            },

        }
        pretrained_state_dict = pretrained_model.state_dict()
        new_state_dict = {}
        for pretrained_key, my_model_key in state_dict_mapping.items():
            if pretrained_key in pretrained_state_dict.keys():
                new_state_dict[my_model_key] = pretrained_state_dict[pretrained_key]
            else:
                raise RuntimeError(f"not found state dict key {pretrained_key} in pretrained model")

        self.load_state_dict(new_state_dict, strict=False)


class BertForClassificationWithHFBertBase(nn.Module):
    def __init__(
        self, num_classes: int, hidden_size: int, check_point_name: str
    ) -> None:
        super().__init__()
        self.bert_base = TransformerBertModel.from_pretrained(check_point_name)
        self.classifier = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.bert_base(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        cls_embeddings = hidden_states.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embeddings)
        return logits


def params_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_forward_pass():
    """Test the forward pass of the BERT model"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create test inputs
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, 30522, (batch_size, seq_length)).to(
        device
    )  # Random token IDs
    token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long).to(
        device
    )  # All type 0
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(
        device
    )  # All tokens active

    # Initialize model
    config = BertForClassifierConfig(num_classes=2, device=device)
    model = BertForClassification(config)
    model.to(device)

    # Run forward pass
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids, attention_mask)

    # Verify output shapes
    assert outputs.shape == (
        batch_size,
        config.num_classes,
    ), f"Expected output shape {(batch_size, config.num_classes)}, got {outputs.shape}"

    print("Forward pass test passed!")


if __name__ == "__main__":
    # Run the tests
    test_forward_pass()
