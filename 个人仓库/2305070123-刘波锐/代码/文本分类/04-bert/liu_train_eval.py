import time

import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from sklearn import metrics
from utils.liu_utils import get_time_dif
import numpy as np

def train(model, train_iter, dev_iter, config):
    device = config.device
    model.to(device)
    # 损失函数
    loss_fn = CrossEntropyLoss().to(device)
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    start_time = time.time()
    dev_best_loss = float("inf")

    # 遍历epoch
    for epoch in range(config.num_epochs):
        # 遍历batch
        total_batch = 0
        for i, (x, y) in enumerate(train_iter):
            # 前向传播
            output = model(x)
            # 损失计算
            loss = loss_fn(output, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 每100个batch输出在训练集和验证集上的效果
            if total_batch % 100 == 0 and total_batch != 0:
                true = y.data.cpu().numpy()
                predict = torch.max(output.data, 1)[1].cpu().numpy()
                train_acc = metrics.accuracy_score(true, predict)
                # 评估验证集效果
                dev_acc, dev_loss = evaluate(config, model, dev_iter, test=False)
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

def evaluate(config, model, dev_iter, test):
    model.eval()

    model = model.to(config.device)


    loss_fn = CrossEntropyLoss().to(config.device)
    total_loss = 0
    predict_all = np.array([])
    label_all = np.array([])

    with torch.no_grad():
        for text, label in dev_iter:
            logits = model(text)
            loss = loss_fn(logits, label)
            total_loss += loss.item()

            predict = torch.max(logits.data, 1)[1].cpu().numpy()
            label = label.cpu().numpy()

            label_all = np.append(label_all, label)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(label_all, predict_all)
    if test:
        report = metrics.classification_report(label_all, predict_all)
        metric = metrics.confusion_matrix(label_all, predict_all)
        return acc, total_loss / len(dev_iter) ,report, metric

    else:
        return acc, total_loss / len(dev_iter)


def test(model, test_iter, config):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loss_fn = CrossEntropyLoss().to(device)

    # 加载最佳模型
    # model.load_state_dict(torch.load(config.save_path))
    # model.to(device)

    acc, loss, report, confusion = evaluate(config, model, test_iter, test=True)

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Loss: {loss:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(confusion)
    print("=" * 60)






