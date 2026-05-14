from attr.validators import max_len
from torch.utils.data import Dataset
import torch
import pickle as pkl


class MyDataset(Dataset):
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_id = self.input_list[idx]
        input_id = input_id[:self.max_len]
        input_id = torch.tensor(input_id, dtype=torch.long)

        return input_id

if __name__ == '__main__':
    with open('medical_train.pkl', 'rb') as f:
        data = pkl.load(f)
    max_len = 36

    dataset = MyDataset(data, max_len)
    print(dataset[0])