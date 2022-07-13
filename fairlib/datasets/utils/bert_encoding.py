import numpy as np
import torch
from transformers import *
import pickle
from tqdm.auto import tqdm

class BERT_encoder:
    def __init__(self) -> None:
        self.model, self.tokenizer = self.load_lm()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

    
    def load_lm(self):
        model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        return model, tokenizer

    def tokenize(self, data):
        """tokenize texts

        Args:
            data (list): a list of strings

        Returns:
            list: list of tokenized data
        """
        tokenized_data = []
        for row in tqdm(data):
            tokens = self.tokenizer.encode(row, add_special_tokens=True)
            # keeping a maximum length of bert tokens: 512
            tokenized_data.append(tokens[:512])
        return tokenized_data

    def encode_text(self, data):
        all_data_cls = []
        all_data_avg = []
        batch = []
        for row in tqdm(data):
            batch.append(row)
            input_ids = torch.tensor(batch).cuda()
            with torch.no_grad():
                last_hidden_states = self.model(input_ids)[0].detach().cpu()
                all_data_avg.append(last_hidden_states.squeeze(0).mean(dim=0).numpy())
                all_data_cls.append(last_hidden_states.squeeze(0)[0].numpy())
            batch = []
        return np.array(all_data_avg), np.array(all_data_cls)

    def encode(self, data):
        tokens = self.tokenize(data)

        avg_data, cls_data = self.encode_text(tokens)

        return avg_data, cls_data