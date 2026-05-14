import torch
import torch.nn as nn

from liu_config import *
from utils.liu_common import *
from utils.liu_dataloder import *

# 定义只有BILSTM实现
class NERBiLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, droupout_p, word2id, tag2id):
        super().__init__()
        self.name = "BiLSTM"
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_size = len(tag2id)

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_dim//2,
                            bidirectional=True,
                            batch_first=True
                            )

        self.dropout = nn.Dropout(droupout_p)

        self.linear = nn.Linear(self.hidden_dim, self.tag_size)

    def forward(self, x, mask):
        embed_x = self.embed(x)
        output, hidden = self.lstm(embed_x)
        output_mask = output * mask.unsqueeze(-1)
        droupout_out = self.dropout(output_mask)

        res = self.linear(droupout_out)
        return res



if __name__ == '__main__':
    config = Config()
    datas, word2id = build_data()
    print(len(word2id))
    ner_lstm = NERBiLSTM(config.embedding_dim, config.hidden_dim, config.dropout, word2id, config.tag2id)
    print(ner_lstm)

    a,b = get_data()
    for x, j, attention in a:
        res = ner_lstm(x, attention)
        print(res.shape)
        break