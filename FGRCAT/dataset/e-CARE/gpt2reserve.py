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

    prompt = """请将下面这个句子用英文改为相反的意思，并且直接输出最终答案就可以！""" + input_text

    inputs = tokenizer.encode(prompt, return_tensors="pt").cuda()

    outputs = model.generate(inputs, max_length=512, num_return_sequences=1, temperature=0.9)

    opposite_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return opposite_sentence


# 加载train.jsonl文件
modified_data = []

count = 0
with open("train.jsonl", "r") as f:
    for line in f:
        count = count + 1
        print(str(count) + "/" + "14928")
        example = json.loads(line)
        original_premise = example["premise"]
        original_label = example["label"]
        new_label = 1 if original_label == 0 else 0

        opposite_sentence = generate_opposite_sentence_chatGLM(original_premise)

        if opposite_sentence:
            # 按句号分割句子，并取倒数第二个句子作为最终句子
            sentences = opposite_sentence.split(".")
            if len(sentences) >= 2:
                opposite_sentence_last_sentence = sentences[-2].strip() + "."
            else:
                opposite_sentence_last_sentence = opposite_sentence.strip()
            opposite_sentence_last_sentence = extract_english(opposite_sentence_last_sentence)
            print(opposite_sentence_last_sentence)


        modified_example = {
            "index": example["index"],
            "premise": opposite_sentence_last_sentence,
            "ask-for": example["ask-for"],
            "hypothesis1": example["hypothesis1"],
            "hypothesis2": example["hypothesis2"],
            "label": new_label
        }
        modified_data.append(modified_example)


with open("train_reversed.jsonl", "w", encoding="utf-8") as f:
    for example in modified_data:
        example_str = json.dumps(example, indent=4, ensure_ascii=False)
        f.write(example_str.rstrip() + '\n')  # 移除结尾换行符并写入对象到文件中


