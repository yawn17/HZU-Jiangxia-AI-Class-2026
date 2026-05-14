
# from model  import bert
from importlib import import_module
import torch

def inference(model, text, config, pad_size=32):
    token = config.tokenizer.tokenize(text)
    tokens = ['[CLS]'] + token

    token_id = config.tokenizer.convert_tokens_to_ids(tokens)
    seq_len = len(token_id)
    if seq_len < pad_size:
        token_id += [0] * (pad_size - seq_len)
        mask = [1] * seq_len + [0] * (pad_size - seq_len)
    else:
        token_id = token_id[:pad_size]
        mask = [1] * pad_size
        seq_len = pad_size

    token_id = torch.LongTensor(token_id).to(config.device)
    mask = torch.LongTensor(mask).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)

    token_id = token_id.unsqueeze(0)
    mask = mask.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)

    data = (token_id, seq_len, mask)
    logits = model(data)
    class_id = torch.max(logits.data, 1)[1]

    return class_id


def id_to_name(class_id, config):
    name = config.class_list[class_id]
    return name

if __name__ == '__main__':
    X = import_module('model.bert')
    config = X.Config()
    model = X.Model(config)
    weight = torch.load(config.save_path, map_location=config.device)
    model.load_state_dict(weight)

    model = model.to(config.device)

    text = '惠州学院倒闭了'
    id = inference(model, text, config, pad_size=32)
    name = id_to_name(id,config)
    print(name)


