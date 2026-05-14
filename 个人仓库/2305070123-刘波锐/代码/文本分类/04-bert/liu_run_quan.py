import torch
from model.liu_bert import Config, Model
from liu_train_eval import test
from utils.liu_utils import build_dataset, build_iter

config = Config()
print(config.device)
model = Model(config)
weight = torch.load(config.save_path, map_location=config.device)

# 将模型架构和训练好的模型权重参数组合在一起
model.load_state_dict(weight)

q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear},dtype=torch.qint8)
# 加载测试集数据
_, test_data, _ = build_dataset(config)
test_data = build_iter(test_data, config)

test(q_model, test_data, config)
