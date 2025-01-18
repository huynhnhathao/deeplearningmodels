from transformers import BertTokenizer
from datasets import load_dataset

if __name__ == "__main__":
    checkpoint_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(checkpoint_name)
    dataset = load_dataset("glue", "sst2")

    print("hello")
