# 模型预测
# from model import bert
from importlib import import_module
import torch


# 定义模型推理函数
def inference(model, text, config, pad_size=32):
    # 文本处理
    # 分词
    tokens = config.tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens
    # id
    token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
    seq_len = len(token_ids)
    # 填充或截断
    if seq_len < pad_size:
        token_ids += [0] * (pad_size - seq_len)
        mask = [1] * seq_len + [0] * (pad_size - seq_len)
    else:
        token_ids = token_ids[:pad_size]
        mask = [1] * pad_size
        seq_len = pad_size
    # 类型和形状
    token_ids = torch.LongTensor(token_ids).to(config.device)
    mask = torch.LongTensor(mask).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    token_ids = token_ids.unsqueeze(0)
    mask = mask.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)
    data = (token_ids, seq_len, mask)
    # 模型推理
    logits = model(data)
    # max
    cls_id = torch.max(logits, 1)[1]
    return cls_id


def id2name(id, config):
    name = config.class_list[id]
    return name


# 加载模型并获取数据
if __name__ == '__main__':
    # 加载训练好的模型
    X = import_module('model.bert')
    config = X.Config()
    model = X.Model(config).to(config.device)
    weight = torch.load(config.save_path, map_location=config.device)
    model.load_state_dict(weight)
    # 获取文本数据
    text = '一体推进教育发展、科技创新、人才培养'
    # 推理函数
    cls_id = inference(model, text, config)
    name = id2name(cls_id, config)
    print(name)
