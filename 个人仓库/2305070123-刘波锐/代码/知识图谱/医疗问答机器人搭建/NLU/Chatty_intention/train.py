# -*- coding:utf-8 -*-
import os
import pickle
import random
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import locale

# 设置区域编码
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

seed = 222
random.seed(seed)
np.random.seed(seed)


def load_data(data_path):
    x, y = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().split(',')
            text_list = ' '.join(list(text))

            x.append(text_list)
            y.append(label)

    index = np.arange(len(x))
    np.random.shuffle(index)
    X = [x[i] for i in index]
    Y = [y[i] for i in index]
    return X, Y


def run(data_path, save_path):
    X, y = load_data(data_path)
    label_set = sorted(list(set(y)))

    label2id = {label: idx for idx, label in enumerate(label_set)}
    id2label = {idx: label for idx, label in enumerate(label_set)}
    y = [label2id[i] for i in y]
    label_names = sorted(label2id.items(), key=lambda x: x[1])
    target_names = [i[0] for i in label_names]
    labels = [i[1] for i in label_names]

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state=42)

    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=0.0, max_df=0.9, analyzer='char')
    train_x = vec.fit_transform(train_x)
    test_x = vec.transform(test_x)

    # 修改 LogisticRegression 的使用方式
    LR = OneVsRestClassifier(LogisticRegression(C=8, dual=False, max_iter=400, random_state=122))
    LR.fit(train_x, train_y)
    preds = LR.predict(test_x)
    print(classification_report(test_y, preds, target_names=target_names))
    print(confusion_matrix(test_y, preds, labels=labels))

    gbdt = GradientBoostingClassifier(n_estimators=450, learning_rate=0.01, max_depth=8, random_state=24)
    gbdt.fit(train_x, train_y)
    pred = gbdt.predict(test_x)
    print(classification_report(test_y, pred, target_names=target_names))
    print(confusion_matrix(test_y, pred, labels=labels))

    lr_pred = LR.predict_proba(test_x)
    gdt_pred = gbdt.predict_proba(test_x)

    pred = np.argmax((lr_pred + gdt_pred) / 2, axis=1)
    print(classification_report(test_y, pred, target_names=target_names))
    print(confusion_matrix(test_y, pred, labels=labels))

    pickle.dump(id2label, open(os.path.join(save_path, 'id2label.pkl'), 'wb'))
    pickle.dump(vec, open(os.path.join(save_path, 'vec.pkl'), 'wb'))
    pickle.dump(LR, open(os.path.join(save_path, 'LR.pkl'), 'wb'))
    pickle.dump(gbdt, open(os.path.join(save_path, 'gbdt.pkl'), 'wb'))


if __name__ == '__main__':
    train_data_path = 'data/train.txt'
    save_path = 'save_model'
    run(train_data_path, save_path)