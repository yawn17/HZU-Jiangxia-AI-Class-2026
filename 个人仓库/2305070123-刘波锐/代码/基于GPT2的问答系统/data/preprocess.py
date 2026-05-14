from transformers import BertTokenizerFast
import pickle
from tqdm import tqdm

def preprocess(train_txt_path, train_pkl_path):

    tokenizer = BertTokenizerFast(r'D:\大三上\09 正课_大模型微调开发(基于GPT-2模型的医疗问诊对话系统)\05-code_edit\liu_03-GPT2\vocab\vocab2.txt',
                                              sep_token='[SEP]',
                                              pad_token='[PAD]',
                                              cls_token='[CLS]',
                                              )

    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id

    with open(train_txt_path, 'rb') as f:
        data = f.read().decode('utf-8')

    # print(data)

    if '\r\n' in data:
        data = data.split('\r\n\r\n')
    else:
        data = data.split('\n\n')

    # print(data)

    sample_id = []

    for idx, sample in enumerate(data):
        if '\r\n' in sample:
            samples = sample.split('\r\n')
        else:
            samples = sample.split('\n')

        input_id = [cls_id]
        for sentence in samples:
            input_id += tokenizer.encode(sentence, add_special_tokens=False)
            input_id += [sep_id]

        sample_id.append(input_id)

    print(len(sample_id))

    with open(pkl_path, 'wb') as f:
        pickle.dump(sample_id, f)




if __name__ == '__main__':
    txt_path = r'D:\大三上\09 正课_大模型微调开发(基于GPT-2模型的医疗问诊对话系统)\05-code_edit\liu_03-GPT2\data\medical_train.txt'
    pkl_path = r'D:\大三上\09 正课_大模型微调开发(基于GPT-2模型的医疗问诊对话系统)\05-code_edit\liu_03-GPT2\data\medical_train.pkl'
    preprocess(txt_path, pkl_path)