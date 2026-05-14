import json

from torch.utils.data import DataLoader, Dataset
from process import *
from config import *
config = Config()

class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.datas = [json.loads(line) for line in open(data_path, encoding="utf-8")]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        content = self.datas[idx]
        text = content['text']
        spo_list = content['spo_list']
        return text, spo_list


def get_data():
    train_dataset = MyDataset(config.train_data_path)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  shuffle=True
                                  )

    # 实例化验证数据集Dataset对象
    dev_data = MyDataset(config.dev_data_path)

    dev_dataloader = DataLoader(dataset=dev_data,
                                batch_size=config.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                drop_last=True)
    # 实例化测试数据集Dataset对象
    test_data = MyDataset(config.test_data_path)

    # 实例化测试数据集Dataloader对象
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 drop_last=True)

    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    # rd = MyDataset(config.train_data_path)
    # print(rd.datas[0])
    # print(rd[0])
    train_dataloader, dev_dataloader, test_dataloader = get_data()
