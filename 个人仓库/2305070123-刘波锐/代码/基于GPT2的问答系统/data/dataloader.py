import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
from data.dataset import MyDataset

def load_dataset(train_pkl_path, val_pkl_path):
    with open(train_pkl_path, 'rb') as f:
        train_input_list = pickle.load(f)

    with open(val_pkl_path, 'rb') as f:
        val_input_list = pickle.load(f)


    train_dataset = MyDataset(train_input_list, 200)
    valid_dataset = MyDataset(val_input_list, 200)
    return train_dataset, valid_dataset

def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)

    return input_ids, labels

def get_dataloader(train_pkl_path, val_pkl_path, batch_size):
    train_dataset, val_dataset = load_dataset(train_pkl_path, val_pkl_path)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn,
                                  drop_last=False
                                  )

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                drop_last=False
                                )

    return train_dataloader, val_dataloader

if __name__ == '__main__':
    train_path = 'medical_train.pkl'
    val_path = 'medical_valid.pkl'
    # train_dataset, valid_dataset = load_dataset(train_path, val_path)
    #
    # print(train_dataset[0])
    # print(valid_dataset[0])

    train_dataloader, val_dataloader = get_dataloader(train_path, val_path)
    for x,y in train_dataloader:
        print(x)
        print(y)
        print(x.shape)
        print(y.shape)
        break