# 设备CPU
import torch
from model.bert import Config, Model
from train_eval import test
from utils.utils import build_dataset,build_iter

# 加载模型
config = Config()
print(config.device)
model = Model(config=config)
print(model)
# 加载模型权重
weight = torch.load(config.save_path, map_location=config.device)
# 将模型架构和权重组合在一起
model.load_state_dict(weight)
# 动态量化
q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print(q_model)
# 测试模型效果
# 加载测试集数据
test_dataset = build_dataset(config)
test_iter = build_iter(test_dataset,config)

# 模型测试
# test(q_model,test_iter,config)
# 保存量化模型权重
# torch.save(q_model,config.save_path2)
