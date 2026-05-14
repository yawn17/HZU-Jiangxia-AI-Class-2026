import torch
import torch.nn as nn
import time
import torch.optim as optim
from model.liu_BiLSTM import *
from model.liu_BiLSTM_CRF import *
from utils.liu_dataloder import *
from  tqdm import tqdm
# classification_report可以导出字典格式，修改参数：output_dict=True，可以将字典在保存为csv格式输出
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from liu_config import *
config = Config()


def model2train():
    train_dataloader, dev_dataloader = get_data()

    models = {
        "BiLSTM":NERBiLSTM,
        "BiLSTM_CRF":NERLSTM_CRF
    }

    model = models[config.model](config.embedding_dim, config.hidden_dim, config.dropout,word2id,config.tag2id)
    model = model.to(config.device)
    print(model)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    start_time = time.time()

    if config.model == "BiLSTM":
        f1_score = -1000
        for epoch in range(config.epochs):
            model.train()
            for index, (input, label, mask) in enumerate(tqdm(train_dataloader, desc='BiLSTM训练')):
                x = input.to(config.device)
                y = label.to(config.device)
                mask = mask.to(config.device)
                logits = model(x, mask)

                pred = logits.view(-1, len(config.tag2id))
                loss = criterion(pred, y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if index % 100 == 0:
                    print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))

            precision, recall, f1, report = model2dev(dev_dataloader, model, criterion)
            print(report)
            if f1 > f1_score:
                f1_score = f1
                torch.save(model.state_dict(), 'model/liu_bilstm_best.pth')
                print(report)
        end_time = time.time()
        print(f'训练总耗时：{end_time - start_time}')
    elif config.model == "BiLSTM_CRF":
        f1_score = -1000
        for epoch in range(config.epochs):
            model.train()
            for index, (input, label, mask) in enumerate(tqdm(train_dataloader, desc='BiLSTM_CRF训练')):
                x = input.to(config.device)
                y = label.to(config.device)
                # mask 需要提前转换成布尔类型
                mask = mask.to(torch.bool).to(config.device)
                loss = model.log_likelihood(x, y, mask).mean()
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
                optimizer.step()
                if index % 100 == 0:
                    print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))

            precision, recall, f1, report = model2dev(dev_dataloader, model, criterion)
            print(report)
            if f1 > f1_score:
                f1_score = f1
                torch.save(model.state_dict(), 'model/liu_bilstm_crf_best.pth')
                print(report)
        end_time = time.time()
        print(f'Bilstm_crf训练总耗时：{end_time - start_time}')



def model2dev(dev_iter, model, criterion=None):
    # cel 损失函数
    avg_loss = 0
    preds = []
    golds = []
    model.eval()
    for index, (input, label, mask) in enumerate(tqdm(dev_iter, desc='BiLSTM验证')):
        x = input.to(config.device)
        y = label.to(config.device)
        mask = mask.to(config.device)

        pred = []
        if model.name == "BiLSTM":
            logits = model(x, mask)
            predict = torch.argmax(logits, dim=-1).tolist()
            pred = logits.view(-1, len(config.tag2id))

            loss = criterion(pred, y.view(-1))
            avg_loss += loss.item()
        elif model.name == "BiLSTM_CRF":
            mask = mask.to(torch.bool).to(config.device)
            predict = model(x,mask)
            loss = model.log_likelihood(x, y, mask).mean()
            avg_loss += loss.item()

        # 获取样本非0元素
        length = []

        for value in input:
            tmp = [i.item() for i in value if i.item() > 0]
            length.append(tmp)

        for index, i in enumerate(predict):
            preds.extend(i[:len(length[index])])

        for index, i in enumerate(label.tolist()):
            golds.extend(i[:len(length[index])])
    precision = precision_score(golds, preds, average='macro')
    recall = recall_score(golds, preds, average='macro')
    f1 = f1_score(golds, preds, average='macro')
    report = classification_report(golds, preds)

    return precision, recall, f1, report


if __name__ == '__main__':
    model2train()