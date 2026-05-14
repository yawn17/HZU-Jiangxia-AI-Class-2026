import time
import torch
import torch.optim as optim
from model import *
from intent_config import *
from flask import Flask, request, jsonify
app = Flask(__name__)
conf = Config()

label_list = [line.strip() for line in open('data/label.txt', 'r', encoding='utf-8')]


id2label = {idx:value for idx, value in enumerate(label_list)}


model = MyModel(bert_path=conf.bert_path, bert_hidden=768, tag_size=conf.num_class)
model.load_state_dict(torch.load('./save_model/epoch_10.pth'))
model = model.to(conf.device)


def model2predict(sample, model):
    # 对数据进行处理
    inputs = tokenizer.encode_plus(sample,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=60,
                                   return_tensors='pt')
    input_ids = inputs["input_ids"].to(conf.device)
    attention_mask = inputs["attention_mask"].to(conf.device)
    token_type_ids = inputs["token_type_ids"].to(conf.device)
    # 将数据送入模型
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
    logits = torch.softmax(logits,dim=-1)
    out = torch.argmax(logits, dim=-1).item()
    value, index= torch.topk(logits, k=1)
    return {"name": id2label[out], "confidence": round(float(value.item()), 3)}

@app.route('/service/api/bert_intent_recognize', methods=['GET','POST'])
def bert_intent_recognize():
    data = {"success": 0}
    res = None
    param = request.get_json()
    text = param['text']
    try:
        res = model2predict(text, model)
        data['res'] = res
        data['success'] = 1
    except:
        print('模型出错')

    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)