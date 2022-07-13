import numpy as np
import torch
from transformers import *
import pickle
from tqdm.auto import tqdm, trange

class BERT_encoder:
    def __init__(self, batch_size=64) -> None:
        self.batch_size = batch_size
        self.model, self.tokenizer = self.load_lm()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

    
    def load_lm(self):
        model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        return model, tokenizer

    def tokenize(self, data):

        tokenized_data = []
        total_n = len(data)
        n_iterations = (total_n // self.batch_size) + (total_n % self.batch_size > 0)
        for i in trange(n_iterations):
            row_lists = list(data)[i*self.batch_size:(i+1)*self.batch_size]
            tokens = self.tokenizer(row_lists, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")['input_ids']
            tokenized_data.append(tokens)
        return tokenized_data

    def encode_text(self, data):
        all_data_cls = []
        all_data_avg = []
        for row in tqdm(data):
            with torch.no_grad():
                input_ids = row.to(self.device)
                last_hidden_states = self.model(input_ids)[0].detach().cpu()
                all_data_avg.append(last_hidden_states.mean(dim=1).numpy())
                all_data_cls.append(last_hidden_states[:,0].numpy())
                input_ids = input_ids.detach().cpu()
        return np.vstack(np.array(all_data_avg)), np.vstack(np.array(all_data_cls))


    def encode(self, data):
        tokens = self.tokenize(data)

        avg_data, cls_data = self.encode_text(tokens)

        return avg_data, cls_data