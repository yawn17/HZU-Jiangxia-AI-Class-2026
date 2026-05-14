import os
import torch
import json


class Config(object):
    def __init__(self):
        # 如果是windows或者linux电脑（使用GPU）
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        # M1芯片及其以上的电脑（使用GPU）
        # self.device = 'mps'
        self.train_path = r'D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\LSTM_CRF\data\train.txt'
        self.vocab_path = r'D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\LSTM_CRF\vocab\vocab.txt'
        self.embedding_dim = 300
        self.epochs = 5
        self.batch_size = 16
        self.hidden_dim = 256
        self.lr = 1e-3 # crf的时候，lr可以小点，比如1e-3
        self.dropout = 0.2
        self.model = "BiLSTM_CRF" # 可以只用"BiLSTM"
        # self.model = "BiLSTM"
        self.tag2id = json.load(open(r'D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\LSTM_CRF\data\tag2id.json'))

if __name__ == '__main__':
    conf = Config()
    print(conf.train_path)
    print(conf.tag2id)
