import torch
import torch.nn as nn
from utils.data_loader import *
from transformers import BertModel


class MyModel(nn.Module):
    def __init__(self, bert_path, bert_hidden, tag_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.linear = nn.Linear(bert_hidden, tag_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        pool_output = self.bert(input_ids, attention_mask, token_type_ids).pooler_output
        output = self.linear(pool_output)
        return output

if __name__ == '__main__':
    train_iter, dev_iter = get_dataloader()
    mymodel = MyModel(conf.bert_path, 768, conf.num_class).to(conf.device)
    for input_ids, attention_mask, token_type_ids, labels in train_iter:
        output = mymodel(input_ids, attention_mask, token_type_ids)
        print(output.shape)
        break
