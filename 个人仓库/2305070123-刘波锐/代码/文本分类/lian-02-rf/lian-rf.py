import jieba
import pandas as pd
from collections import Counter

df = pd.read_csv(r'D:\大三上\07 正课 头条投满分项目-V6-25年7月版本-8天-AI版本\05-code_edit\01-data\dev.txt', sep='\t', names=['text', 'label'])

# print(df.head())

count = Counter(df.label.values)

# print(count)
# 1.2 样本分析
total = sum(count.values())
for i, v in count.items():
    print(i, v / total * 100,'%')

df['len'] = df['text'].apply(len)
print(df.head())
len_mean = df['len'].mean()
print(len_mean)

len_std = df['len'].std()
print(len_std)

# 统计信息，设置seq_len mean+std,mean+2std,mean+3std

# 1.3 jieba分词
def cut_sentence(s):
    return jieba.lcut(s)

df['word'] = df['text'].apply(lambda x: ' '.join(cut_sentence(x))[:30])
print(df.head())

df.to_csv('./dev.csv')




























