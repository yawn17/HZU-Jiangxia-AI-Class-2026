import fasttext

model = fasttext.train_supervised(input='train_fast1.txt', # 训练集路径
                                  autotuneValidationFile='dev_fast1.txt', # 验证集路径
                                  autotuneDuration=120, # 时间单位为s
                                  wordNgrams=2, # N_gram
                                  verbose=3)

res = model.test('dev_fast1.txt')
print(res)