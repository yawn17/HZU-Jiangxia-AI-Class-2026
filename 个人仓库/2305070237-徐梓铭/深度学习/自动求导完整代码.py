import torch
def get_hanshu():
    """获取函数表达式"""
    hanshu = input("输入函数（例如:x**3+4*x**2+8*x+5):")
    def get_eval(x):
        return eval(hanshu, {"x": x})
    return get_eval

def get_x():
    """获取求导的x值"""
    x = float(input("输入x的值((例如:3):"))
    return x

def hanshuqiudao(hanshu, x):
    """函数求导"""
    x1 = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y = hanshu(x1)
    y.backward()
    return x1.grad.item()

# 主程序
if __name__ == "__main__":
    hanshu = get_hanshu()
    x = get_x()
    result = hanshuqiudao(hanshu, x)
    print(f"导数值：{result}")