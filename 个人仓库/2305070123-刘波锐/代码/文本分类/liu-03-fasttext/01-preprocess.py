
id_to_name = {}
id = 0
import jieba

with open(r'D:\大三上\07 正课 头条投满分项目-V6-25年7月版本-8天-AI版本\05-code_edit\01-data\class.txt','r',encoding='utf-8') as f1:
    for line in f1:
        class_name = line.strip('\n').strip()
        id_to_name[id] = class_name
        id += 1

print(id_to_name)

data = []

with open(r'D:\大三上\07 正课 头条投满分项目-V6-25年7月版本-8天-AI版本\05-code_edit\01-data\train.txt','r',encoding='utf-8') as f2:
    for line in f2:
        line = line.strip('\n').strip()
        text, label = line.split('\t')

        id = int(label)
        label_name = id_to_name[id]
        label_name = '__label__' + label_name

        # 按词划分
        # text = ' '.join(jieba.lcut(text))
        # 按字划分
        text = ' '.join(list(text))

        line = label_name + ' ' + text
        data.append(line)

print(data[:5])
with open('train_fast1.txt','w',encoding='utf-8') as f3:
    for line in data:
        f3.write(line+'\n')

