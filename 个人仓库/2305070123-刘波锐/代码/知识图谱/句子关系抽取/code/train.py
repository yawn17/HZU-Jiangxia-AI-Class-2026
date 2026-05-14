# coding:utf-8
from model.CasrelModel import *
from utils.process import *
from utils.data_loader import *
from config import *
import pandas as pd
from tqdm import tqdm

def model2train(model, train_iter, dev_iter, optimizer, config):
    epochs = config.epochs
    best_triple_f1 = 0
    for epoch in range(epochs):
        best_triple_f1 = train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch)


def train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch):
    for step, (input, label) in enumerate(tqdm(train_iter, desc='Casrel训练')):
        model.train()
        logits = model(**input)
        loss = model.compute_loss(**logits, **label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step  % 1500 == 0:
            # torch.save(model.state_dict(),
            #            '../save_model/epoch_%s_model_%s.pth' % (epoch, step))
            results = model2dev(model, dev_iter)
            torch.save(model.state_dict(), '../save_model/best_f1.pth')
            print(results[-1])
            if results[-2] > best_triple_f1:
                best_triple_f1 = results[-2]
                print('epoch:{},'
                      'step:{},'
                      'sub_precision:{:.4f}, '
                      'sub_recall:{:.4f}, '
                      'sub_f1:{:.4f}, '
                      'triple_precision:{:.4f}, '
                      'triple_recall:{:.4f}, '
                      'triple_f1:{:.4f},'
                      'train loss:{:.4f}'.format(epoch,
                                                 step,
                                                 results[0],
                                                 results[1],
                                                 results[2],
                                                 results[3],
                                                 results[4],
                                                 results[5],
                                                 loss.item()))
    return best_triple_f1


def model2dev(model, dev_iter):
    model.eval()
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)
    print(f'df.shape: {df}')
    for inputs, labels in tqdm(dev_iter):
        logist = model(**inputs)
        pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        batch_size = inputs['input_ids'].shape[0]
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

        for batch_idx in range(batch_size):
            pred_subs = extract_sub(pred_sub_heads[batch_idx].squeeze(),pred_sub_tails[batch_idx].squeeze())
            true_subs = extract_sub(sub_heads[batch_idx].squeeze(),
                                    sub_tails[batch_idx].squeeze())
            pred_objs = extract_obj_and_rel(pred_obj_heads[batch_idx],
                                            pred_obj_tails[batch_idx])

            true_objs = extract_obj_and_rel(obj_heads[batch_idx],
                                            obj_tails[batch_idx])
            df['PRED']['sub'] += len(pred_subs)
            df['REAL']['sub'] += len(true_subs)

            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df['TP']['sub'] += 1

            df['PRED']['triple'] += len(pred_objs)
            df['REAL']['triple'] += len(true_objs)

            for true_obj in true_objs:
                if true_obj in pred_objs:
                    df['TP']['triple'] += 1

    # 计算指标
    df.loc['sub','p'] = df.loc['sub','TP'] / (df.loc['sub','PRED'] + 1e-9)
    df.loc['sub', 'r'] =  df.loc['sub','TP'] / (df.loc['sub','REAL'] + 1e-9)

    df.loc['sub','f1'] = 2 * df.loc['sub','p'] * df.loc['sub','r'] / (df.loc['sub','p'] + df.loc['sub', 'r'] + 1e-9)

    sub_precision = df.loc['sub','p']
    sub_recall = df.loc['sub','r']
    sub_f1 = df.loc['sub','f1']

    df.loc['triple', 'p'] = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    df.loc['triple', 'r'] = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    df.loc['triple', 'f1'] = 2 * df['p']['triple'] * df['r']['triple'] / (
            df['p']['triple'] + df['r']['triple'] + 1e-9)

    triple_precision = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    triple_recall = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (
            triple_precision + triple_recall + 1e-9)

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


if __name__ == '__main__':
    config = Config()
    model, optimizer, sheduler, device = load_model(config)
    train_dataloader, dev_dataloader, test_dataloader = get_data()
    model2train(model, train_dataloader, dev_dataloader, optimizer, config)