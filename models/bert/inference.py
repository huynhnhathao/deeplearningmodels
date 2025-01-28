from models.bert import BertTokenizer, BertForClassification

if __name__ == "__main__":
    checkpoint_name = "bert-base-uncased"
    tokenizer = BertTokenizer(checkpoint_name)
