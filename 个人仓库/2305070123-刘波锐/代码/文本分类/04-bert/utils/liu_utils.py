from model.bert import Config
import tqdm
import torch
import time
from datetime import timedelta


def build_dataset(config):

    def load_dataset(path, pad_size=32):
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                text, label = line.split('\t')

                tokens = config.tokenizer.tokenize(text)
                tokens = ['[cls]'] + tokens
                tokens_ids = config.tokenizer.convert_tokens_to_ids(tokens)
                seq_len = len(tokens)
                if seq_len > pad_size:
                    tokens_ids = tokens_ids[:pad_size]
                    mask = [1] * pad_size
                    seq_len = pad_size
                else:
                    mask = [1] * seq_len + [0] * (pad_size - seq_len)
                    tokens_ids += [0] * (pad_size - seq_len)
                dataset.append([tokens_ids, int(label), seq_len, mask])

        return dataset

    train = load_dataset(config.train_path,config.pad_size)
    test = load_dataset(config.test_path,config.pad_size)
    dev = load_dataset(config.dev_path,config.pad_size)

    return train, test, dev


class DataIter(object):
    def __init__(self, dataset, batch_size, device, model_name):
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = device
        self.model_name = model_name
        self.index = 0
        # 批次数量
        self.n_batchs = len(self.dataset) // self.batch_size
        self.residue = False
        if len(self.dataset) % self.batch_size != 0:
            self.residue = True

    def __iter__(self):
        return self

    def to_tensor(self, batch_data):
        x = torch.LongTensor([_[0] for _ in batch_data]).to(self.device)
        y = torch.LongTensor([_[1] for _ in batch_data]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in batch_data]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in batch_data]).to(self.device)
        return (x, seq_len, mask), y


    def __next__(self):
        if self.residue and self.index == self.n_batchs:
            batch_data = self.dataset[self.index * self.batch_size : len(self.dataset)]
            batch_data = self.to_tensor(batch_data)
            self.index += 1
            return batch_data
        elif self.index >= self.n_batchs:
            self.index = 0
            raise StopIteration
        else:
            batch_data = self.dataset[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            batch_data = self.to_tensor(batch_data)
            self.index += 1
            return batch_data


    def __len__(self):
        if self.residue:
            return self.n_batchs + 1
        else:
            return self.n_batchs


def get_time_dif(start_time):
    """
    计算已使用的时间差。
    """
    # 获取当前时间
    end_time = time.time()
    # 计算时间差
    time_dif = end_time - start_time
    # 将时间差转换为整数秒，并返回时间差对象
    return timedelta(seconds=int(round(time_dif)))


def build_iter(dataset, config):
    iter = DataIter(dataset=dataset, batch_size=config.batch_size, device=config.device, model_name=config.model_name)
    return iter


if __name__ == '__main__':
    config = Config()
    train, test, dev=  build_dataset(config)
    # print(len(train))
    # print(len(test))
    # print(len(dev))
    iter = build_iter(train, config)
    for data in iter:
        print(data)
