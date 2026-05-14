#损失函数实战
import torch
import torch.nn as nn
#模拟预测值
pred=torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
#真实值
true=torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
true_diff=torch.tensor([[1.5,2.5],[3.5,4.5],[5.5,6.5]])

#分类标签
label=torch.tensor([0,1,0])
l1=nn.L1Loss()
loss1=l1(pred,true_diff)
loss2=l1(pred,true)
print(f"L1Loss平均绝对误差，求损失值：{loss1.item()}")
print(loss2.item())

mse=nn.MSELoss()
loss3=mse(pred,true_diff)
print(f"MSEWLoss(均分误差）求损失值：{loss3}")

ce=nn.CrossEntropyLoss()
loss4=ce(pred,label)
print(f"CrossEntropyLoss(交叉熵)求损失值：{loss4}")

kl=nn.KLDivLoss(reduction='batchmean')
pred_log = torch.log_softmax(pred, dim=-1)
true_soft = torch.softmax(true, dim=-1)
loss5 = kl(pred_log, true_soft)
print(" KLDivLoss (KL散度 - 分布比较的损失值 ：", loss5.item())