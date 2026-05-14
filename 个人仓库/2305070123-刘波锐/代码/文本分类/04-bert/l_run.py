from importlib import import_module
from utils.liu_utils import build_dataset, build_iter, get_time_dif
from liu_train_eval import train, test


if __name__ == '__main__':
    model_name = 'bert'

    try:
        # 动态导入模型模块
        model_module = import_module(f'model.{model_name}')
        config = model_module.Config()

        print(f"Using device: {config.device}")
        print(f"Model save path: {config.save_path}")

        # 数据准备
        train_dataset, test_dataset, dev_dataset = build_dataset(config)
        train_iter = build_iter(train_dataset, config)
        test_iter = build_iter(test_dataset, config)
        dev_iter = build_iter(dev_dataset, config)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(dev_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        # 模型初始化
        model = model_module.Model(config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 训练和测试
        train(model, train_iter, dev_iter, config)
        test(model, test_iter, config)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
