import json
import os
from collections import Counter
os.chdir('..')
cur = os.getcwd()
print('当前数据处理默认工作目录：', cur)

print(os.path.join(cur, 'data/labels.json'))

class TransferData():
    def __init__(self):
        self.labels_dict = json.load(open(os.path.join(cur, 'data/labels.json'),'r',encoding='utf-8'))
        self.seq_tag_dict = json.load(open(os.path.join(cur, 'data/tag2id.json'),'r',encoding='utf-8'))
        self.origin_path = os.path.join(cur, 'data_origin')
        self.train_filepath = os.path.join(cur, 'data/train.txt')

    def transfer(self):
        with open(self.train_filepath,'w',encoding='utf-8') as fr:
            for root, dirs, files in os.walk(self.origin_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    if 'original' not in filepath:
                        continue
                    label_filepath = filepath.replace('.txtoriginal','')
                    res_dict = self.read_label_txt(label_filepath)
                    with open(filepath, encoding='utf-8') as f:
                        content = f.read().strip()
                        for idx, char in enumerate(content):
                            char_label = res_dict.get(idx, 'O')
                            fr.write(char + '\t' + char_label + '\n')


    def read_label_txt(self, label_filepath):
        res_dict = {}
        with open(label_filepath,'r',encoding='utf-8') as f:
            for line in f.readlines():
                res = line.strip().split('\t')
                start = int(res[1])
                end = int(res[2])
                label = res[3]
                label_tag = self.labels_dict.get(label)
                for i in range(start, end+1):
                    if i == start:
                        tag = 'B-' + label_tag
                    else:
                        tag = 'I-' + label_tag
                    res_dict[i] = tag
        return res_dict


if __name__ == '__main__':
    td = TransferData()
    td.transfer()
