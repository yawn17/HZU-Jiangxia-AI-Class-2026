import json
import torch
from utils.liu_common import *
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from liu_config import Config

config = Config()
datas, word2id = build_data()
# print(datas)
# print(word2id)

# 1 构建Datas类
class NERDataset(Dataset):
    def __init__(self, datas):
        super().__init__()
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        x = self.datas[idx][0]
        y = self.datas[idx][1]
        return x, y

def collate_fn(batch):
    x_train = [torch.tensor([word2id[word] for word in data[0]]) for data in batch]
    y_train = [torch.tensor([config.tag2id[tag] for tag in data[1]]) for data in batch]

    # 对x_train 进行长度补齐,根据每个批次的最大长度进行补齐
    x_tensor = pad_sequence(x_train, batch_first=True, padding_value=0)
    y_tensor = pad_sequence(y_train, batch_first=True, padding_value=0)
    print(x_tensor.shape)
    attention_mask = (x_tensor !=0).long()

    return x_tensor, y_tensor, attention_mask

def get_data():
    train_dataset = NERDataset(datas[:6200])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True)


    dev_dataset = NERDataset(datas[6200:])
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                shuffle=True
                                )
    return train_dataloader, dev_dataloader


if __name__ == '__main__':
    # nd = NERDataset(datas)
    # print(nd[2])
    a,b = get_data()
    for i, j, k in a:
        print(i.shape, j.shape, k.shape)
        print(k)

        break
