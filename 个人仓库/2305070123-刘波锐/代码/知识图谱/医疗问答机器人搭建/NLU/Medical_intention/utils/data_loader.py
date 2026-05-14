import os
import torch
import pandas as pd
import numpy as np
# from intent_config import *
from intent_config import Config
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

conf = Config()
tokenizer = BertTokenizer.from_pretrained(conf.bert_path)


def load_data(path):
    df = pd.read_csv(path,header=0, sep=',', names=["text", "label_class", "label_id"])
    texts = df.text.to_list()
    labels = df.label_id.map(int).to_list()
    return texts, labels

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.texts, self.labels = load_data(data_path)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        return text, label

def collate_fn(datas):
    batch_text = [item[0] for item in datas]
    batch_label = [item[1] for item in datas]
    inputs = tokenizer.batch_encode_plus(batch_text,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=conf.max_len,
                                         return_tensors='pt')
    input_ids = inputs["input_ids"].to(conf.device)
    attention_mask = inputs["attention_mask"].to(conf.device)
    token_type_ids = inputs["token_type_ids"].to(conf.device)
    labels = torch.tensor(batch_label, dtype=torch.long, device=conf.device)
    return input_ids, attention_mask, token_type_ids, labels

def get_dataloader():
    train_dataset = MyDataset(conf.train_path)
    train_iter = DataLoader(train_dataset,
                            batch_size=conf.batch_size,
                            collate_fn=collate_fn,
                            drop_last=True,
                            shuffle=True)
    dev_dataset = MyDataset(conf.train_path)
    dev_iter = DataLoader(dev_dataset,
                          batch_size=conf.batch_size,
                          collate_fn=collate_fn,
                          drop_last=True,
                          shuffle=True)
    return train_iter, dev_iter


if __name__ == '__main__':
    train_iter, dev_iter = get_dataloader()
    for input_ids, attention_mask, token_type_ids, labels in train_iter:

        break
