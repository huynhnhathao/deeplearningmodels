from dataclasses import dataclass
import torch
from torch.nn import functional as F
import torch.nn as nn

import math


@dataclass
class BertConfig:
    hidden_dim: int = 768
    max_sequence_length: int = 512
    vocab_size: int = 30522
    type_vocab_size: int = 2
    padding_token_id: int = 0

    # self-attention parameters
    num_heads: int = 12
    embedding_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    attention_linear_dropout: float = 0.1
    fcnn_middle_dim: int = 3072
    position_wise_ffnn_dropout: float = 0.1

    num_encoder_layers: int = 1
    layer_norm_eps: float = 1e-12


@dataclass
class BertForClassifierConfig(BertConfig):
    num_classes: int = 10


class BertEmbedding(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.hidden_dim
        )
        self.token_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_dim
        )
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.padding_token_id
        )
        # the original transformers encoder applies dropout to the sum of embeddings and positional embeddings
        self.dropout = nn.Dropout(config.embedding_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_dim, config.layer_norm_eps)

    def forward(
        self,
        token_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        token_ids, token_type_ids, position_ids shape: (batch_size, sequence_length)
        token_ids: vocab indices of the tokens
        token_type_ids: either 0 or 1 per position in the sequence, use to differentiate two sentences
        position_ids: count from 0 at the start of the sequence to max_sequence_length
        """
        embeddings = (
            self.embedding(token_ids)
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
            config.attention_dropout_prob,
        )

        self.self_attention_layer_norm = nn.LayerNorm(
            config.hidden_dim, config.layer_norm_eps, bias=True
        )

        self.fcnn = FFNN(
            config.hidden_dim, config.fcnn_middle_dim, config.position_wise_ffnn_dropout
        )

        self.fcnn_layer_norm = nn.LayerNorm(config.hidden_dim, config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.self_attention_layer_norm(
            hidden_states
            + self.multi_head_self_attention(hidden_states, attention_mask)
        )
        hidden_states = self.fcnn_layer_norm(hidden_states + self.fcnn(hidden_states))
        return hidden_states


class FFNN(nn.Module):
    def __init__(self, hidden_dim: int, middle_dim: int, dropout_prob: float) -> None:
        super().__init__()
        self.intermediate = nn.Linear(hidden_dim, middle_dim, bias=True)
        self.output = nn.Linear(middle_dim, hidden_dim, bias=True)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(self.gelu(self.intermediate(hidden_states)))
        hidden_states = self.dropout(self.gelu(self.output(hidden_states)))
        return hidden_states


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attention_dropout_prob: float,
    ):
        """
        Transformer Self-attention
            Args
                - hidden_dim: size of the token embedding and hidden_state
                - num_heads: number of attention head
                - attention_dropout_prob: dropout probability apply to the attention scores,
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

        # because hidden_dim % num_heads == 0,
        # output dim for num_heads is also the embedding_dim
        self.Q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.K = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.V = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.attention_dropout = nn.Dropout(attention_dropout_prob)
        self.attention_output = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attention_output_dropout = nn.Dropout(config.attention_linear_dropout)

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

        # attention_scores shape (batch_size, num_heads, sequence_length, sequence_length)
        # swap keys to have shape (batch_size, self.num_heads, self.dk, sequence_length)
        attention_scores = torch.matmul(queries, torch.transpose(keys, -1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        # attention mask is of shape (batch_size, sequence_length),
        # need to create a tensor of shape (batch_size, num_heads, sequence_length, sequence_length)
        # to fit the attention_scores tensor
        # the element-wise logical end operation is to create an attention square matrix such that
        # padded tokens can't be attended to and can not attend to other tokens
        attention_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(
            attention_mask.unsqueeze(1) == 0, float("-inf")
        )
        attention_scores = F.softmax(attention_scores, -1)

        # apply dropout to the attention scores, random tokens will not be attended to at training
        attention_scores = self.attention_dropout(attention_scores)

        # attention_values shape: (batch_size, num_heads, sequence_length, dk)
        attention_values = torch.matmul(attention_scores, values)

        # concat all heads' values to one vector representation per token
        # (batch_size, sequence_length, hidden_dim)
        # this is the "concat then linear projection step" in the Attention is all you need paper
        attention_values = attention_values.view(batch_size, sequence_length, -1)
        return self.attention_output_dropout(self.attention_output(attention_values))


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )

    def forward(self, hidden_states, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        """
        Extract the first token embedding vector out of the hidden_states tensor
        then apply a linear and a tanh function to it
        """
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
        self, token_ids, token_type_ids, position_ids, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Input tensors to this module is expected to be already padded to the model's max_sequence_length
        """
        self.validate_input_dimension(
            token_ids, token_type_ids, position_ids, attention_mask
        )

        # input_ids shape: (batch_size, sequence_length)
        # sequence_length must match the max_sequence_length, this module expects all padding be done before it
        embedding = self.embedding(token_ids, token_type_ids, position_ids)
        # hidden_states shape (batch_size, sequence_length, hidden_dim)
        hidden_states = self.encoder(embedding, attention_mask)
        return hidden_states

    def validate_input_dimension(
        self, token_ids, token_type_ids, position_ids, attention_mask: torch.Tensor
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


class BertForClassification(nn.Module):
    def __init__(self, config: BertForClassifierConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.classifier_head = nn.Linear(config.hidden_dim, config.num_classes)
        self.padding_token_id = config.padding_token_id
        self.max_sequence_length = config.max_sequence_length

        # for classification task the model's input is one sentence hence token_type_ids only contains one id
        self.token_type_ids = torch.zeros(
            (1, self.max_sequence_length), dtype=torch.long
        )
        self.position_ids = torch.arange(
            0, self.max_sequence_length, dtype=torch.long
        ).view(1, -1)

    def forward(self, token_ids, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, _ = token_ids.size()

        hidden_states = self.bert(
            self.pad_sequence(token_ids, self.padding_token_id),
            self.token_type_ids.expand(batch_size, -1),
            self.position_ids.expand(batch_size, -1),
            self.pad_sequence(attention_mask, 0),
        )
        # use the first token's embedding (the CLS token) for classification
        cls_embedding = hidden_states[:, 0, :]

        return self.classifier_head(cls_embedding)

    def pad_sequence(self, input_ids: torch.Tensor, fill_value: int) -> torch.Tensor:
        # right-padding the input tensor with the padding index to match the max_sequence_length
        # input_ids has shape (batch_size, sequence_length)
        batch_size, sequence_length = input_ids.size()
        if sequence_length < self.max_sequence_length:
            padding = torch.full(
                (batch_size, self.max_sequence_length - sequence_length), fill_value
            )
            return torch.cat((input_ids, padding), dim=1)

        return input_ids

    def load_model_weights(self, model_name_or_path: str) -> None:
        """
        Load weights of a model identifier from huggingface's model hub
        """
        pass


def params_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = BertForClassifierConfig()
    model = BertForClassification(config)
    token_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 125))
    attention_mask = torch.full((2, 125), 1)
    probs = model(token_ids, attention_mask)
    print(probs.shape)
