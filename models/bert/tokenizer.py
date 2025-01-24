from typing import List
import torch
from typing import Dict, List, Optional, Union
import re

from transformers import AutoTokenizer

from models.bert.model import BertForClassification


class BertTokenizer:
    def __init__(
        self,
        model_name_or_path: str,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
        lower_case: bool = True,
    ):
        """
        Initialize BERT tokenizer with vocabulary and special tokens.

        Args:
            vocab: Dictionary mapping tokens to their ids
            unk_token: Unknown token string
            pad_token: Padding token string
            cls_token: Classification token string
            sep_token: Separator token string
            mask_token: Mask token string
            lower_case: Whether to lowercase input text
        """
        self.vocab = self.load_vocab(model_name_or_path)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.lower_case = lower_case

        # Create reverse mapping
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Precompile regex patterns
        self.word_piece_pattern = re.compile(r"[\w']+|[^\w\s]")
        self.whitespace_pattern = re.compile(r"\s+")

    def tokenize(self, text: str) -> List[str]:
        """Convert text to list of tokens"""
        if self.lower_case:
            text = text.lower()

        text = self.whitespace_pattern.sub(" ", text).strip()
        # Split text into words and punctuation
        tokens = []
        for word in self.word_piece_pattern.findall(text):
            if word in self.vocab:
                tokens.append(word)
            else:
                # Handle unknown words with wordpiece algorithm
                subwords = self._tokenize_sub_word(word)
                tokens.extend(subwords)
        return tokens

    def _tokenize_sub_word(self, word: str) -> list[str]:
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            while start < end:
                subword = word[start:end]
                if start > 0:
                    subword = "##" + subword
                if subword in self.vocab:
                    tokens.append(subword)
                    start = end
                    break
                end -= 1
            else:
                tokens.append(self.unk_token)
                break
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to their corresponding ids"""
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of ids to their corresponding tokens"""
        return [self.inv_vocab.get(id, self.unk_token) for id in ids]

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Union[Dict[str, torch.Tensor], Dict[str, List[List[int]]]]:
        """
        Main tokenization method that handles single or batch text processing.

        Args:
            text: Input text or list of texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Return type ('pt' for PyTorch tensors)

        Returns:
            Dictionary containing:
            - input_ids: Token ids
            - token_type_ids: Segment ids (always 0 for single sequence)
            - attention_mask: Attention mask (1 for real tokens, 0 for padding)
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize all texts
        batch_tokens = [self.tokenize(t) for t in text]

        # Add special tokens
        batch_tokens = [
            [self.cls_token] + tokens + [self.sep_token] for tokens in batch_tokens
        ]

        # Convert to ids
        input_ids = [self.convert_tokens_to_ids(tokens) for tokens in batch_tokens]

        # Handle truncation
        if truncation and max_length:
            input_ids = [ids[:max_length] for ids in input_ids]

        # Handle padding
        max_len = max(len(ids) for ids in input_ids)
        attention_mask = []
        if padding:
            attention_mask = [
                [1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids
            ]

            input_ids = [
                ids + [self.vocab[self.pad_token]] * (max_len - len(ids))
                for ids in input_ids
            ]
        else:
            attention_mask = [
                [1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids
            ]

        # Create token type ids (all zeros for single sequence)
        token_type_ids = [[0] * len(ids) for ids in input_ids]

        # Convert to tensors if requested
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids),
                "token_type_ids": torch.tensor(token_type_ids),
                "attention_mask": torch.tensor(attention_mask),
            }

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

    def load_vocab(self, model_name_or_path: str) -> Dict[str, int]:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return tokenizer.vocab


def test_tokenizer():
    text = [
        "I am trying to implement the BERT tokenizer from scratch",
        "This type of learning is fascinating",
    ]
    tokenizer = BertTokenizer("bert-base-uncased")
    tokenized = tokenizer(text, return_tensors="pt")
    print(tokenized)


if __name__ == "__main__":
    test_tokenizer()
