import torch
import torch.nn as nn
from TorchCRF import CRF
from liu_config import *
from utils.liu_common import *
from utils.liu_dataloder import *

class NERLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NERLSTM_CRF, self).__init__()
        self.name = "BiLSTM_CRF"
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_to_ix = tag2id
        self.tag_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        #CRF
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.crf = CRF(self.tag_size)

    def forward(self, x, mask):
        output = self.get_lstm2linear(x)
        output = output * mask.unsqueeze(-1)
        result = self.crf.viterbi_decode(output, mask)
        return  result

    def log_likelihood(self, x, tags, mask):
        output = self.get_lstm2linear(x)
        output = output * mask.unsqueeze(-1)
        # 送入模型计算损失
        return - self.crf(output, tags, mask)

    def get_lstm2linear(self, x):
        embedding = self.word_embeds(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return outputs

if __name__ == '__main__':
    config = Config()
    datas, word2id = build_data()
    ner_lstm_crf = NERLSTM_CRF(config.embedding_dim, config.hidden_dim, config.dropout, word2id, config.tag2id)
    print(ner_lstm_crf)

    a,b = get_data()
    for x, y, attention in a:
        mask = attention.to(torch.bool)
        res = ner_lstm_crf.log_likelihood(x, y, mask)
        print(res)
        break