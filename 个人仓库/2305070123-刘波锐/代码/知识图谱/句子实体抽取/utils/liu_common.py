from liu_config import *
config = Config()

def build_data():
    data = []
    sample_x = []
    sample_y = []
    vocab_list = ['PAD','UNK']
    with open(config.train_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word_tag = line.rstrip().split('\t')
            if not word_tag:
                continue
            word = word_tag[0]
            if not word:
                continue
            tag = word_tag[-1]
            sample_x.append(word)
            sample_y.append(tag)
            if word not in vocab_list:
                vocab_list.append(word)
            if word in ['。', '?', '!', '！', '？']:
                data.append([sample_x, sample_y])
                sample_x = []
                sample_y = []
    word2id = {word:index for index, word in enumerate(vocab_list)}
    write_vocab(vocab_list, config.vocab_path)
    return data, word2id

def write_vocab(words_list, filename):
    with open(filename, 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(words_list))

if __name__ == '__main__':
    data, word2id = build_data()
    print(len(data))
    print(data)
    print(word2id)
