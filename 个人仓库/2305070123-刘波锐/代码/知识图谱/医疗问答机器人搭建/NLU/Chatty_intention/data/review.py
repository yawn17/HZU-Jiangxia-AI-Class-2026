import numpy as np
# X = [10, 3, 40, 5]
# # a = sorted(X)
# X.sort()
# print(X)
# index = np.arange(len(X))
# print(index)
# np.random.shuffle(index)
# print(index)
# #
dict1 = {"salary": 200, "jiangjin": 20, "buzhu": 40}
print(dict1.items())
print(sorted(dict1.items(), key=lambda x: x[1],  reverse=True))

#
# text = '早上好呀'
# print(list(text))
