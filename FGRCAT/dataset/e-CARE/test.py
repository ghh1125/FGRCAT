import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

# 定义正则表达式，用于匹配英文句子
english_pattern = re.compile(r'[a-zA-Z\s,;.!?]+')

path = "/home/amax/ghh/chatglm2/ChatGLM2"  # ChatGLM模型的路径

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).cuda()


def extract_english(text):
    pattern = re.compile(r'([a-zA-Z.,!? ]+)')
    matches = pattern.findall(text)

    english_text = ''.join(matches)
    return english_text.strip()


def generate_opposite_sentence_chatGLM(input_text):
    prompt = """请你选出正确的选项并且输出！""" + input_text

    inputs = tokenizer.encode(prompt, return_tensors="pt").cuda()

    outputs = model.generate(inputs, max_length=512, num_return_sequences=1, temperature=0.9)

    opposite_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return opposite_sentence


# 加载train.jsonl文件
modified_data = []

count = 0
with open("dev.jsonl", "r") as f:
    for line in f:
        count = count + 1
        if count == 10:
            break
        print(str(count) + "/" + "14928")
        example = json.loads(line)
        question = example["premise"] + "answer:" + " A:" + example["hypothesis1"] + "or  B:" + example["hypothesis2"]
        answer = generate_opposite_sentence_chatGLM(question)
        print(answer)

        modified_example = {
            "label": answer
        }
        modified_data.append(modified_example)

with open("answer.jsonl", "w", encoding="utf-8") as f:
    for example in modified_data:
        example_str = json.dumps(example, indent=4, ensure_ascii=False)
        f.write(example_str.rstrip() + '\n')  # 移除结尾换行符并写入对象到文件中
