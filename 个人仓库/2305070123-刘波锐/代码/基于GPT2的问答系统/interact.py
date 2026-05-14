import os
from datetime import datetime
from transformers import GPT2LMHeadModel
from transformers import BertTokenizerFast
import torch
import torch.nn.functional as F
# 注意：确保 parameter_config.py 文件存在且 ParameterConfig 类定义正确
from parameter_config import *

config = ParameterConfig()
PAD = '[PAD]'
pad_id = 0


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now
    top_k = min(top_k, logits.size(-1))  # Safety check
    print(f'top_k---->{top_k}')
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        print(f'sorted_logits-->{sorted_logits}')
        print(f'sorted_indices-->{sorted_indices}')
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained('./save_model/bj_epoch10')
    model = model.to(config.device)
    model.eval()

    # 加载分词器
    tokenizer = BertTokenizerFast(
        r'D:\大三上\09 正课_大模型微调开发(基于GPT-2模型的医疗问诊对话系统)\05-code_edit\liu_03-GPT2\vocab\vocab2.txt',
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
    )

    samples_file = None  # 初始化文件句柄
    try:
        # 处理日志保存路径（修复路径拼接问题）
        if config.save_samples_path:
            # 确保路径结尾有目录分隔符
            save_path = os.path.join(config.save_samples_path, '')  # 补充分隔符
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # 使用 os.path.join 拼接路径，避免分隔符问题
            samples_file_path = os.path.join(save_path, 'samples.txt')
            # 追加模式打开（避免每次运行清空），如果要清空可以用 'w'
            samples_file = open(samples_file_path, 'a', encoding='utf-8')
            samples_file.write(f"聊天记录{datetime.now()}:\n")
            samples_file.flush()  # 立即刷入磁盘

        history = []
        print('开始和chatbot聊天，输入CTRL + C以退出（Windows下CTRL+Z可能无效，建议用CTRL+C）')

        while True:
            text = input("请输入：")  # 增加提示，提升体验
            if not text.strip():  # 处理空输入
                print("输入不能为空，请重新输入！")
                continue

            print(f"你输入的内容：{text}")
            # 写入用户输入并刷新
            if samples_file:
                samples_file.write(f"user:{text}\n")
                samples_file.flush()

            # 编码输入文本
            input_id = tokenizer.encode(text, add_special_tokens=False)
            print(f'input_id-->{input_id}')

            history.append(input_id)
            input_id = [tokenizer.cls_token_id]

            # 拼接历史对话
            for history_utr in history[-config.max_history_len:]:
                input_id.extend(history_utr)
                input_id.append(tokenizer.sep_token_id)

            # 转换为tensor并添加batch维度
            input_id = torch.tensor(input_id).long().to(config.device)
            input_id = input_id.unsqueeze(0)

            response = []
            for _ in range(config.max_len):
                with torch.no_grad():  # 推理阶段禁用梯度计算，节省内存
                    outputs = model(input_id)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]

                # 重复惩罚
                for id in set(response):
                    next_token_logits[id] /= config.repetition_penalty

                # 修复：[UNK] 拼写错误（少了]）
                unk_token_id = tokenizer.convert_tokens_to_ids('[UNK]')
                next_token_logits[unk_token_id] = -float('Inf')

                # 过滤logits
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits,
                    top_k=config.topk,
                    top_p=config.topp,
                    filter_value=-float('Inf')
                )

                # 采样下一个token
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                if next_token == tokenizer.sep_token_id:
                    break

                response.append(next_token.item())
                input_id = torch.cat((input_id, next_token.unsqueeze(0)), dim=1)

            # 解码回复并打印
            history.append(response)
            response_text = tokenizer.convert_ids_to_tokens(response)
            response_text = "".join(response_text)
            print(f"chatbot:{response_text}")

            # 写入机器人回复并刷新
            if samples_file:
                samples_file.write(f"chatbot:{response_text}\n")
                samples_file.flush()

    except KeyboardInterrupt:
        print("\n退出聊天...")
    except Exception as e:
        # 捕获所有异常，避免程序直接崩溃导致日志丢失
        print(f"程序出现异常：{e}")
    finally:
        # 无论是否异常，最终都关闭文件
        if samples_file:
            samples_file.close()
            print("日志文件已保存并关闭")


if __name__ == '__main__':
    main()