from transformers import AutoTokenizer

class text2id():
    """mapping natural language to numeric identifiers.
    """
    def __init__(self, args) -> None:
        if args.encoder_architecture == "Fixed":
            self.encoder = None
        elif args.encoder_architecture == "BERT":
            self.model_name = 'bert-base-cased'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            raise NotImplementedError
    
    def encoder(self, sample):
        encodings = self.tokenizer(sample, truncation=True, padding=True)
        return encodings["input_ids"]
