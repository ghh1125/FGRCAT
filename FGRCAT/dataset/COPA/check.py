import json

def check_json_format(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()  # 读取文件的第一行并去除末尾的换行符
        if first_line.startswith('{') and first_line.endswith('}'):
            try:
                json.loads(first_line)  # 尝试解析第一行内容为JSON对象
                print("第一行内容符合JSON格式要求。")
            except json.JSONDecodeError as e:
                print("第一行内容不是有效的JSON格式：", e)
        else:
            print("第一行内容不是以 '{' 开始或以 '}' 结束，不符合JSON格式的要求。")

# 指定要检查的文件路径
file_path = "devEG.jsonl"

# 调用函数进行检查
check_json_format(file_path)
