# 执行的主函数
from importlib import import_module
from utils.utils import build_dataset, build_iter, get_time_dif
from train_eval import train,test

if __name__ == '__main__':
    model_name = "bert"
    if model_name == "bert":
        X = import_module('model.bert')
        config = X.Config()
        print(config.device)
        # 数据集获取
        # train_data, test_data, dev_data = build_dataset(config)
        dev_data = build_dataset(config)
        # train_iter = build_iter(train_data, config)
        # test_iter = build_iter(test_data, config)
        dev_iter = build_iter(dev_data, config)
        # 模型实例化
        model = X.Model(config)
        # 模型训练
        train(model,dev_iter,dev_iter,config)
        # 模型验证
        test(model,dev_iter,config)

    else:
        pass
