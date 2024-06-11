import json

def convert_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    converted_data = []
    for line in data:
        json_obj = json.loads(line)
        index = "train-" + str(json_obj["idx"])
        premise = json_obj["premise"]
        ask_for = json_obj["question"]
        hypothesis1 = json_obj["choice1"]
        hypothesis2 = json_obj["choice2"]
        label = json_obj["label"]

        converted_obj = {
            "index": index,
            "premise": premise,
            "ask-for": ask_for,
            "hypothesis1": hypothesis1,
            "hypothesis2": hypothesis2,
            "label": label
        }

        converted_data.append(converted_obj)

    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in converted_data:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

convert_format('test.zh.jsonl', 'train.jsonl')
# convert_format('val.zh.jsonl', 'dev.jsonl')
