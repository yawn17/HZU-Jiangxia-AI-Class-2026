# coding:utf-8
import torch
import torch.nn as nn
from transformers import BertModel, AdamW
from config import *
config = Config()
from data_loader import *


class Casrel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 加载bert预训练模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.sub_heads_linear = nn.Linear(config.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(config.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(config.bert_dim, config.num_rel)
        self.obj_tails_linear = nn.Linear(config.bert_dim, config.num_rel)

    def get_encoder_text(self,input_ids, mask):
        encoder_text = self.bert(input_ids=input_ids, attention_mask=mask)[0]
        return encoder_text

    def get_subs(self, encoder_text):
        sub_head = torch.sigmoid(self.sub_heads_linear(encoder_text))
        sub_tail = torch.sigmoid(self.sub_tails_linear(encoder_text))
        return sub_head, sub_tail

    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoder_text):
        sub = torch.matmul(sub_head2tail, encoder_text)
        # 平均信息,主语信息可能比较短，除以主语长度平均一下信息
        sub_len = sub_len.unsqueeze(1)
        sub_avg = sub / sub_len
        encoder_text = sub_avg + encoder_text
        # 以上就是将主语信息加到整个句子里面
        obj_heads = torch.sigmoid(self.obj_heads_linear(encoder_text))
        obj_tails = torch.sigmoid(self.obj_tails_linear(encoder_text))
        return obj_heads, obj_tails


    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        # [32,70]
        encoder_text = self.get_encoder_text(input_ids, mask)
        # print(f'encoder_text: {encoder_text.shape}')
        # 2
        sub_head_pred, sub_tail_pred= self.get_subs(encoder_text) # [32, 70, 768] -> [32,70,1]
        # print(f'sub_head_pred: {sub_head_pred.shape}')
        # print(f'sub_tail_pred: {sub_tail_pred.shape}')
        # 3
        sub_head2tail = sub_head2tail.unsqueeze(1) # [32,70] -> [32,1, 70]
        obj_heads_pred, obj_tails_pred = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoder_text)
        # print(f'obj_heads_pred: {obj_heads_pred.shape}')
        # print(f'obj_tails_pred: {obj_tails_pred.shape}')
        result_dict = {'pred_sub_heads': sub_head_pred,
                       'pred_sub_tails': sub_tail_pred,
                       'pred_obj_heads': obj_heads_pred,
                       'pred_obj_tails': obj_tails_pred,
                       'mask': mask}
        return result_dict

    def compute_loss(self,pred_sub_heads, pred_sub_tails,
                     pred_obj_heads, pred_obj_tails,
                     mask,
                     sub_heads, sub_tails,
                     obj_heads, obj_tails):
        '''
                计算损失
                :param pred_sub_heads:[16, 200, 1]
                :param pred_sub_tails:[16, 200, 1]
                :param pred_obj_heads:[16, 200, 18]
                :param pred_obj_tails:[16, 200, 18]
                :param mask: shape-->[16, 200]
                :param sub_heads: shape-->[16, 200]
                :param sub_tails: shape-->[16, 200]
                :param obj_heads: shape-->[16, 200, 18]
                :param obj_tails: shape-->[16, 200, 18]
                :return:
                '''
        # todo:sub_heads.shape,sub_tails.shape, mask-->[16, 200]
        # todo:obj_heads.shape,obj_tails.shape-->[16, 200, 18]
        rel_count = obj_heads.size(-1)
        rel_mask = mask.unsqueeze(-1).repeat(1,1,rel_count)
        loss1 = self.loss(pred_sub_heads, sub_heads, mask)
        loss2 = self.loss(pred_sub_tails, sub_tails, mask)
        loss3 = self.loss(pred_obj_heads, obj_heads, rel_mask)
        loss4 = self.loss(pred_obj_tails, obj_tails, rel_mask)
        return loss1 + loss2 + loss3 + loss4

    def loss(self, pred, gold, mask):
        pred = pred.squeeze(-1)
        # 使用BCEloss
        cur_loss = nn.BCELoss(reduction='none')(pred, gold)
        my_loss = torch.sum(cur_loss * mask) / torch.sum(mask)
        return my_loss

def load_model(conf):
    device = conf.device
    model = Casrel(conf)
    model.to(device)
    # 因为本次模型借助BERT做fine_tuning， 因此需要对模型中的大部分参数进行L2正则处理防止过拟合，包括权重w和偏置b
    # prepare optimzier
    # named_parameters()获取模型中的参数和参数名字
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]  # no_decay中存放不进行权重衰减的参数{因为bert官方代码对这三项免于正则化}
    # any()函数用于判断给定的可迭代参数iterable是否全部为False，则返回False，如果有一个为True，则返回True
    # 判断param_optimizer中所有的参数。如果不在no_decay中，则进行权重衰减;如果在no_decay中，则不进行权重衰减
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.learning_rate)
    # 是否需要对bert进行warm_up。这里默认不进行
    sheduler = None

    return model, optimizer, sheduler, device

# if __name__ == '__main__':
    # model = Casrel(config).to(config.device)
    # train_dataloader, dev_dataloader, test_dataloader = get_data()
    # for input, label in train_dataloader:
    #     res = model(**input)
    #     my_loss = model.compute_loss(**res, **label)
    #     print(f'my_loss: {my_loss}')
    #     break
    # load_model(config)