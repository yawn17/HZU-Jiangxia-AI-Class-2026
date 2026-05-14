from random import choice

import torch

from config import *
config = Config()
# print(config.train_data_path)
from collections import defaultdict


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i:i + target_len] == target:
            return i
    return -1


def create_label(inner_triples, inner_input_ids, seq_len):
    inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
    inner_obj_heads = torch.zeros((seq_len, config.num_rel))
    inner_obj_tails = torch.zeros((seq_len, config.num_rel))
    inner_sub_head2tails = torch.zeros(seq_len)
    inner_sub_len = torch.tensor([1], dtype=torch.float)
    s2ro_map = defaultdict(list)
    for inner_triple in inner_triples:
        inner_triple = (
            config.tokenizer(inner_triple['subject'], add_special_tokens=False)['input_ids'],
            config.rel_vocab.to_index(inner_triple['predicate']),
            config.tokenizer(inner_triple['object'], add_special_tokens=False)['input_ids']
        )
        # print(f'inner_triple: {inner_triple}')
        sub_head_idx = find_head_idx(inner_input_ids, inner_triple[0])
        obj_head_idx = find_head_idx(inner_input_ids, inner_triple[2])
        if sub_head_idx != -1 and obj_head_idx != -1:
            sub = (sub_head_idx, sub_head_idx + len(inner_triple[0]) - 1)
            s2ro_map[sub].append((obj_head_idx, obj_head_idx + len(inner_triple[2]) - 1, inner_triple[1]))
    # print(f's2ro_map: {s2ro_map}')
    if s2ro_map:
        for s in s2ro_map:
            inner_sub_heads[s[0]] = 1
            inner_sub_tails[s[1]] = 1
        sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
        inner_sub_head2tails[sub_head_idx:sub_tail_idx + 1] = 1
        inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
        for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
            inner_obj_heads[ro[0]][ro[2]] = 1
            inner_obj_tails[ro[1]][ro[2]] = 1
    return inner_sub_len, inner_sub_head2tails, inner_sub_heads, inner_sub_tails, inner_obj_heads, inner_obj_tails

'''
inner_sub_len = [3]                    # 主语长度
inner_sub_head2tails = [0,1,1,1,0,0,0,0,0]  # 主语内部标记
inner_sub_heads = [0,1,0,0,0,0,0,0,0]       # 主语开始位置  
inner_sub_tails = [0,0,0,1,0,0,0,0,0]       # 主语结束位置
inner_obj_heads = [                        # 宾语开始(按关系)
    [0,0,0,0,...],  # 关系0
    [0,0,0,0,...],  # 关系1  
    [0,0,0,0,0,0,1,0,0],  # 关系2(歌手)在位置6
    ... 
]
inner_obj_tails = [                        # 宾语结束(按关系)
    [0,0,0,0,...],  # 关系0
    [0,0,0,0,...],  # 关系1
    [0,0,0,0,0,0,0,1,0],  # 关系2(歌手)在位置7
    ...
]
'''

def collate_fn(batch):
    # print(f'batch : {batch}')
    text_list = [data[0] for data in batch]
    triple = [data[1] for data in batch]
    # print(f'text_list : {text_list}')
    # print(f'triple_list : {triple}')
    text = config.tokenizer.batch_encode_plus(text_list, padding=True)
    # print(f'text : {text}')

    batch_size = len(text['input_ids']) # 每个batch_zize因为可能最后抛弃最后一个batch
    seq_len = len(text['input_ids'][0]) # 求出这个batch的最长长度
    # print(batch_size)
    # print(seq_len)
    sub_heads = []
    sub_tails = []
    obj_heads = []
    obj_tails = []
    sub_len = []
    sub_head2tail = []
    for batch_idx in range(batch_size):
        inner_inout_ids = text['input_ids'][batch_idx]
        inner_triples = triple[batch_idx]
        results = create_label(inner_triples, inner_inout_ids, seq_len)
        sub_len.append(results[0])
        sub_head2tail.append(results[1])
        sub_heads.append(results[2])
        sub_tails.append(results[3])
        obj_heads.append(results[4])
        obj_tails.append(results[5])
    #
    input_ids = torch.tensor(text['input_ids']).to(config.device)
    mask = torch.tensor(text['attention_mask']).to(config.device)
    sub_heads = torch.stack(sub_heads).to(config.device)
    sub_tails = torch.stack(sub_tails).to(config.device)
    sub_len = torch.stack(sub_len).to(config.device)
    sub_head2tail = torch.stack(sub_head2tail).to(config.device)
    obj_heads = torch.stack(obj_heads).to(config.device)
    obj_tails = torch.stack(obj_tails).to(config.device)

    inputs = {
        'input_ids': input_ids,
        'mask': mask,
        'sub_head2tail': sub_head2tail,
        'sub_len': sub_len
    }
    labels = {
        'sub_heads': sub_heads,
        'sub_tails': sub_tails,
        'obj_heads': obj_heads,
        'obj_tails': obj_tails
    }

    return inputs, labels



def convert_score_to_zero_one(tensor):
    '''
       以0.5为阈值，大于0.5的设置为1，小于0.5的设置为0
    '''
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor

def extract_sub(pred_sub_heads, pred_sub_tails):
    subs = []
    heads = torch.arange(0, len(pred_sub_heads), device=config.device)[pred_sub_heads==1]
    tails = torch.arange(0, len(pred_sub_tails), device=config.device)[pred_sub_tails==1]
    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))

    return subs

def extract_obj_and_rel(obj_heads, obj_tails):
    obj_heads = obj_heads.T
    obj_tails = obj_tails.T
    rel_count = obj_heads.shape[0]
    obj_and_rels = []

    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]
        objs = extract_sub(obj_head, obj_tail)
        if objs:
            for obj in objs:
                start_idx, end_idx = obj
                obj_and_rels.append((rel_index, start_idx, end_idx))

    return obj_and_rels


