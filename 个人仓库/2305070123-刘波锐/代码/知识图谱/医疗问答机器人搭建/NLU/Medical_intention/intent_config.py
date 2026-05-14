import torch

class Config():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_path = r'D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\liu_Medical_KG\NLU\Medical_intention\data\train.csv'
        self.test_path = r'D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\liu_Medical_KG\NLU\Medical_intention\data\test.csv'
        self.label_path = r'D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\liu_Medical_KG\NLU\Medical_intention\data\label.txt'
        self.epochs = 10
        self.lr = 2e-5
        self.batch_size = 16
        self.max_len = 60
        self.num_class = 13
        self.bert_path = r'D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\relationship_extra\bert-base-chinese'


if __name__ == '__main__':
    config = Config()
    print(config.train_path)
    print(config.test_path)
    print(config.label_path)
    print(config.epochs)
    print(config.lr)
    print(config.bert_path)
