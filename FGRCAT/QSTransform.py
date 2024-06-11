import json

# 读取原始数据集
with open("./QA/train_rand_split.jsonl", "r", encoding="utf-8") as f:
    original_data = f.readlines()

# 转换数据格式
converted_data = []
for idx, line in enumerate(original_data):
    instance = json.loads(line)
    index = f"train-{idx}"
    premise = instance["question"]["stem"]
    hypotheses = [choice["text"] for choice in instance["question"]["choices"]]
    label = ord(instance["answerKey"]) - ord('A')  # 将A、B、C、D等转换为0、1、2、3等
    converted_instance = {
        "index": index,
        "premise": premise,
        "ask-for": "cause",
        "hypothesis1": hypotheses[0],
        "hypothesis2": hypotheses[1],
        "hypothesis3": hypotheses[2],
        "hypothesis4": hypotheses[3],
        "hypothesis5": hypotheses[4],
        "label": label
    }
    converted_data.append(converted_instance)

# 将转换后的数据写入新文件
with open("./QA/train.jsonl", "w", encoding="utf-8") as f:
    for instance in converted_data:
        json.dump(instance, f, ensure_ascii=False)
        f.write('\n')
