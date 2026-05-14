# 导入工具包
import requests
import time

# url + data
url = 'http://127.0.0.1:5000/v1/main_server/'
data = {'uid':'123','text':'广东省惠州学院装空调了'}
t1 = time.time()
# 发送请求
res =requests.post(url,data=data)
t2 = time.time()

# 输出结果
print(res.text)
print(t2-t1)