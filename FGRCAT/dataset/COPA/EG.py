import json

def convert_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()


    converted_data = []
    for line in data:
        obj = json.loads(line.strip())  # 去除末尾的换行符并解析 JSON 对象
        index = obj["index"]
        # label = obj["label"]
        conceptual_explanation = obj["conceptual_explanation"]
        cause = obj["cause"]
        # if label == 0:
        #     cause = obj["hypothesis1"]
        # elif label == 1:
        #     cause = obj["hypothesis2"]
        effect = obj["effect"]

        converted_obj = {
            "index": index,
            "cause": cause,
            "effect": effect,
            "conceptual_explanation": conceptual_explanation
        }

        converted_data.append(converted_obj)

    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in converted_data:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

convert_format('devEG.jsonl', 'devEG.jsonl')
