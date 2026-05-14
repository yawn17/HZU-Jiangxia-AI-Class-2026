import time

import torch
import torch.optim as optim
from model import *
from tqdm import tqdm
from utils.data_loader import *

def model2train():
    # 获取数据
    train_iter, dev_iter = get_dataloader()
    # 实例化模型
    my_model = MyModel(conf.bert_path, 768, conf.num_class)
    my_model = my_model.to(conf.device)
    # 实例化优化器
    my_optim = optim.Adam(my_model.parameters(), lr=conf.lr)
    # 实例化损失函数对象
    criation = nn.CrossEntropyLoss()

    # 定义模型训练参数
    my_model.train()
    total_num = 0 # 训练样本参数
    total_loss = 0 # 训练样本的损失
    total_acc = 0 # 预测正确样本的个数

    # 开始训练
    start_time = time.time()
    for epoch_idx in range(conf.epochs):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(train_iter, desc='训练集')):
            outputs = my_model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            my_loss = criation(outputs, labels)
            # 梯度清零
            my_optim.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_optim.step()

            # 打印训练日志
            total_num = total_num + outputs.size(0)
            total_loss = total_loss + my_loss
            acc_num = sum(torch.argmax(outputs, dim=-1) == labels).item()
            total_acc = total_acc + acc_num
            if i % 100 == 0:
                avg_loss = total_loss / total_num
                avg_acc = total_acc / total_num
                use_time = time.time() - start_time
                print(f'当前训练轮次%.d, 平均损失%.2f, 平均准确率%.2f, 耗时%.2f'% (epoch_idx+1, avg_loss, avg_acc, use_time))
    torch.save(my_model.state_dict(), './save_model/epoch_%s.pth'%(epoch_idx+1))
    end_time = time.time()
    print(f'总耗时-->{end_time-start_time}')

if __name__ == '__main__':
    model2train()
