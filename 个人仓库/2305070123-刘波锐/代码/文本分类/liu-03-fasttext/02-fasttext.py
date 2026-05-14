import fasttext

model = fasttext.train_supervised(input = 'train_fast.txt',wordNgrams = 2)
print(model)
print(len(model.words))

res = model.test('dev_fast.txt')
print(res)

# model.save_model('fasttext_model2_jieba.bin')