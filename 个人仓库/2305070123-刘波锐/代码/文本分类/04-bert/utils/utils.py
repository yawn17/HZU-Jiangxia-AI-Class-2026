# 数据处理与构建
# 数据处理
from model.bert import Config
from tqdm import tqdm
import torch
import time
from datetime import timedelta


# 方法:训练集/测试集/验证集(id,label,seq_len,mask)
def build_dataset(config):
    # load_dataset:
    def load_dataset(path, pad_size=32):
        datas = []
        # 读取数据:open
        with open(path, 'r', encoding='utf-8') as f:
            # 遍历每一条数据
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                # 获取文本和标签
                text, label = line.split('\t')
                # 处理文本
                # 分词
                tokens = config.tokenizer.tokenize(text)
                # 添加[CLS] 并转成id
                tokens = ['[CLS]'] + tokens
                token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
                seq_len = len(tokens)
                # 填充或截断.Mask
                mask = []
                if seq_len < pad_size:
                    # 填充
                    mask = [1] * seq_len + [0] * (pad_size - seq_len)
                    token_ids += [0] * (pad_size - seq_len)
                else:
                    # 截断
                    token_ids = token_ids[:pad_size]
                    mask = [1] * pad_size
                    seq_len = pad_size
                datas.append([token_ids, int(label), seq_len, mask])
        return datas

    # train = load_dataset(config.train_path, config.pad_size)
    # test = load_dataset(config.test_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)

    # return train, test, dev
    return dev


# 获取batch_size数据
class DataIter(object):
    def __init__(self, data, batch_size, device, model_name):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        # 批次索引
        self.index = 0
        # 批次数量
        self.n_batches = len(self.data) // self.batch_size
        self.residue = False
        if len(self.data) % self.batch_size != 0:
            self.residue = True

    def __iter__(self):
        return self

    def __to_tensor(self, batch_data):
        x = torch.LongTensor([_[0] for _ in batch_data]).to(self.device)
        y = torch.LongTensor([_[1] for _ in batch_data]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in batch_data]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in batch_data]).to(self.device)
        return (x,seq_len,mask),y

    def __next__(self):
        # 获取一个batch的数据
        if self.residue and self.index == self.n_batches:
            batch_data = self.data[self.index * self.batch_size:len(self.data)]
            batch_data = self.__to_tensor(batch_data)
            self.index += 1
            return batch_data
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batch_data = self.data[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            batch_data = self.__to_tensor(batch_data)
            self.index += 1
            return batch_data

    def __len__(self):
        # 返回迭代的次数
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iter(dataset, config):
    iter = DataIter(data=dataset, batch_size=config.batch_size, device=config.device, model_name=config.model_name)
    return iter

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


if __name__ == '__main__':
    config = Config()
    # train ,test, dev=build_dataset(config)
    dev = build_dataset(config)
    # print(dev)
    iter = build_iter(dev,config)
    for data in iter:
        print(data)
        break
