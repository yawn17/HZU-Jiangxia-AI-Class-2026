import torch
import torch.nn as nn
import torch.optim as optim

from LSTM_CRF.model.liu_BiLSTM import NERBiLSTM
from model.liu_BiLSTM import *
from model.liu_BiLSTM_CRF import *
from tqdm import tqdm

from liu_config import *
config = Config()

from utils.liu_common import *
_, word2id = build_data()

models = {'BiLSTM': NERBiLSTM,
          'BiLSTM_CRF': NERLSTM_CRF}
model = models[config.model](config.embedding_dim, config.hidden_dim, config.dropout, word2id, config.tag2id)
model.load_state_dict(torch.load('save_model/liu_bilstm_crf_best.pth'))

print(model)

id2tag = {values:key for key, values in config.tag2id.items()}
print(id2tag)

def model2test(sample:str):
    x = []
    for token in sample:
        if token not in word2id:
            token = 'UNK'
        x.append(word2id[token])
    print(x)
    x_tensor = torch.tensor([x], dtype=torch.long)
    mask = (x_tensor != 0).long()
    model.eval()
    with torch.no_grad():
        if model.name == 'BiLSTM':
            logits = model(x_tensor, mask)
            print(logits.shape)
            predict = torch.argmax(logits, dim=-1)[0]
            output = [id2tag[i.item()] for i in predict]
            print(output)
        elif model.name == 'BiLSTM_CRF':
            predict = model(x_tensor, mask)
            print(predict)
            output = [id2tag[i] for i in predict[0]]
            print(output)
        chars = [char for char in sample]
        assert len(chars) == len(output)
        res = extract_entities(chars, output)
        return res

def extract_entities(chars, tags):
    entities = []
    entity = []
    entity_type = None
    for word, tag in zip(chars, tags):
        if tag.startswith('B-'):
            if entity:
                entities.append(entity_type, ''.join(entity))
                entity = []
            entity_type = tag.split('-')[1]
            entity.append(word)
        elif tag.startswith('I-') and entity:
            entity.append(word)
        else:
            if entity:
                entities.append((entity_type, ''.join(entity)))
                entity = []
                entity_type = None

    if entity:
        entities.append((entity_type, ''.join(entity)))

    return {entity: entity_type for entity_type, entity in entities}

if __name__ == '__main__':
    result = model2test(sample='小明的父亲患有冠心病及糖尿病，无手术外伤史及药物过敏史')
    print(result)