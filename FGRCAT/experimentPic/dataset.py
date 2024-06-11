import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# 数据集路径
dataset_paths = {
    "e-CARE": "./datasets/e-CARE",
    "COPA": "./datasets/COPA",
    "commonsenseQA": "./datasets/QA"
}

# 初始化选项计数器和总数计数器
option_counts = {dataset_name: {} for dataset_name in dataset_paths}
total_counts = {dataset_name: 0 for dataset_name in dataset_paths}

# 预定义所有可能的选项
all_options = ["0", "1", "2", "3", "4"]  # CommonsenseQA 是5分类任务

# 遍历每个数据集
for dataset_name, dataset_path in dataset_paths.items():
    # 遍历每个数据集的文件
    for filename in ["dev.jsonl", "train.jsonl"]:
        file_path = os.path.join(dataset_path, filename)

        # 打开JSONL文件
        with open(file_path, "r", encoding="utf-8") as file:
            # 逐行读取JSON对象
            for line in file:
                data = json.loads(line)
                label = str(data["label"])  # 确保标签是字符串形式

                # 更新选项计数
                if label in option_counts[dataset_name]:
                    option_counts[dataset_name][label] += 1
                else:
                    option_counts[dataset_name][label] = 1

                # 更新总数计数器
                total_counts[dataset_name] += 1

# 提取选项和比例
options_dict = OrderedDict((option,
                            [option_counts[dataset_name].get(option, 0) / total_counts[dataset_name] for dataset_name in
                             dataset_paths]) for option in all_options)
dataset_names = list(dataset_paths.keys())
options = list(options_dict.keys())
proportions = np.array(list(options_dict.values()))

# 定义自定义颜色
custom_colors = ['#DEEBF7', '#E7E6E6', '#FDF0E9', '#D6DCE5', '#F2F2F2']

# 绘制柱状图
fig, ax = plt.subplots()
width = 0.2
x = np.arange(len(dataset_names))

for i, option in enumerate(options):
    ax.bar(x + i * width, proportions[i], width, label=f"Option {option}", color=custom_colors[i])

ax.set_xlabel('Datasets')
ax.set_ylabel('Proportion')
ax.set_title('Proportion of Options in Different Datasets')
ax.set_xticks(x + width * (len(options) - 1) / 2)
ax.set_xticklabels(dataset_names)
ax.legend()

plt.tight_layout()
plt.show()
