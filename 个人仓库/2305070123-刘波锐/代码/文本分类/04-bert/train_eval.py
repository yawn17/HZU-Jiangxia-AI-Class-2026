# 训练/验证/测试
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AdamW
import torch
from sklearn import metrics
from utils.utils import get_time_dif
import time


def train(model, train_iter, dev_iter, config):
    start_time = time.time()
    # 损失函数
    loss_fn = CrossEntropyLoss()
    # 优化器
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01
         },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         }]
    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')
    # 遍历epoch
    for epoch in range(config.num_epochs):
        # 遍历batch
        for i, (x, labels) in enumerate(train_iter):
            # 前向
            output = model(x)
            # 损失
            loss = loss_fn(output, labels)
            # 清零
            optimizer.zero_grad()
            # 反向
            loss.backward()
            # step
            optimizer.step()
            # 验证保存
            # 每100个batch输出在训练集和验证集上的效果
            # if total_batch % 1 == 0 and total_batch != 0:
            if total_batch % 1 == 0:
                true = labels.data.cpu()
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                # 评估验证集效果
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 若验证集损失更低，保存模型参数
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                else:
                    improve = ""
                # 计算时间差
                time_dif = get_time_dif(start_time)
                # 输出训练和验证集上的效果
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 评估完成后将模型置于训练模式, 更新参数
                model.train()
            # 每个batch结束后累加计数
            total_batch += 1
            break


def evaluate(config, model, dev_iter, test=False):
    model.eval()
    total_loss = 0
    predict_all = np.array([])
    label_all = np.array([])
    with torch.no_grad():
        # 遍历数据
        for i, (text, label) in enumerate(dev_iter):
            # 获取预测logits
            logits = model(text)
            # 计算损失
            loss = CrossEntropyLoss()(logits, label)
            total_loss += loss.item()
            # acc: 最大值所在的索引
            # 获取label信息
            labels = label.data.cpu().numpy()
            # 获取预测结果
            predict = torch.max(logits.data, 1)[1].cpu().numpy()
            label_all = np.append(label_all, labels)
            predict_all = np.append(predict_all, predict)
            if i == 1:
                break
    acc = metrics.accuracy_score(label_all, predict_all)

    # test: report  matrix
    if test:
        report = metrics.classification_report(label_all, predict_all, target_names=config.class_list, digits=4)
        matrix = metrics.confusion_matrix(label_all, predict_all)
        return acc, total_loss / len(dev_iter), report, matrix
    else:
        return acc, total_loss / len(dev_iter)


def test(model, test_iter, config):
    acc, loss, report, matrix = evaluate(config, model, test_iter, test=True)
    print(acc, loss)
    print(report)
    print(matrix)
