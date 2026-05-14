# coding:utf-8
import torch

from model.CasrelModel import *
from utils.process import *
from config import *
conf = Config()

def load_model(model_path):
    model = Casrel(config).to(config.device)
    model.load_state_dict(torch.load(model_path), strict=False)

    return model

def get_inputs(sample, model):
    print(f'sample: {sample}')
    text = config.tokenizer(sample)
    print(f'text: {text}')
    input_ids = torch.tensor([text['input_ids']]).to(conf.device)
    mask = torch.tensor([text['attention_mask']]).to(conf.device)
    seq_len =len(text['input_ids'])
    inner_sub_head2tail = torch.zeros(seq_len)
    sub_len = torch.tensor([1], dtype=torch.float)
    model.eval()
    with torch.no_grad():
        encode_text = model.get_encoder_text(input_ids,mask)
        pred_sub_head, pred_sub_tail = model.get_subs(encode_text)
        # 将预测的结果01转换
        pred_sub_head = convert_score_to_zero_one(pred_sub_head)
        pred_sub_tail = convert_score_to_zero_one(pred_sub_tail)

        pred_subs = extract_sub(pred_sub_head.squeeze(), pred_sub_tail.squeeze())
        if pred_subs:
            sub_head_idx = pred_subs[0][0]
            sub_tail_idx = pred_subs[0][1]
            inner_sub_head2tail[sub_head_idx : sub_tail_idx + 1] = 1
            inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx], dtype=torch.float)
        sub_len = inner_sub_len.unsqueeze(0).to(conf.device)
        sub_head2tail = inner_sub_head2tail.unsqueeze(0).to(conf.device)
        inputs = {'input_ids': input_ids,
                  'mask': mask,
                  'sub_head2tail': sub_head2tail,
                  'sub_len': sub_len}
        return inputs, model


def model2predict(sample, model):
    with open(config.rel_dict_path, 'r', encoding='utf-8') as f:
        id2rel = json.load(f)
    print(f'id2rel: {id2rel}')
    inputs, model = get_inputs(sample, model)
    logist = model(**inputs)
    pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
    pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
    pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
    pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])
    new_dict = {}
    spo_list = []
    ids = inputs['input_ids'][0]
    text_list = config.tokenizer.convert_ids_to_tokens(ids)
    sentence = ''.join(text_list[1: -1])
    pred_subs = extract_sub(pred_sub_heads[0].squeeze(), pred_sub_tails[0].squeeze())
    pred_objs = extract_obj_and_rel(pred_obj_heads[0], pred_obj_tails[0])
    if len(pred_subs) == 0 or len(pred_objs) == 0:
        print('没有识别出结果')
        return {}
    if len(pred_objs) > len(pred_subs):
        pred_subs = pred_subs * len(pred_objs)
    for sub, rel_obj in zip(pred_subs, pred_objs):
        sub_spo = {}
        sub_head, sub_tail = sub
        sub = ''.join(text_list[sub_head: sub_tail + 1])
        if '[PAD]' in sub:
            continue
        sub_spo['subject'] = sub
        relation = id2rel[str(rel_obj[0])]
        obj_head, obj_tail = rel_obj[1], rel_obj[2]
        obj = ''.join(text_list[obj_head: obj_tail + 1])
        if '[PAD]' in obj:
            continue
        sub_spo['predicate'] = relation
        sub_spo['object'] = obj
        spo_list.append(sub_spo)
    new_dict['text'] = sentence
    new_dict['spo_list'] = spo_list
    return new_dict



if __name__ == '__main__':
    sample = '《认真的雪》是薛之谦演唱的歌曲'
    model_path = r'D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\relationship_extra\save_model\last_model_old.pth'
    model = load_model(model_path)
    res = model2predict(sample, model)
    print(f'res: {res}')