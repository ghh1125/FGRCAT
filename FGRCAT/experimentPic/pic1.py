import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(model_paths, model_names):
    plt.figure(figsize=(12, 8))

    for idx, (model_path, model_name) in enumerate(zip(model_paths, model_names), 1):
        with open(model_path, 'r') as file:
            data = json.load(file)

        predictions = [entry["predictions"] for entry in data]
        true_labels = [entry["true"] for entry in data]

        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, predictions)

        # 绘制混淆矩阵
        plt.subplot(2, 2, idx)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=np.unique(true_labels),
                    yticklabels=np.unique(true_labels))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {model_name}')

    plt.tight_layout()
    plt.show()


# 文件路径和模型名称
model_paths = [
    "experiment/e-CARE/增强后/bart+CI+Con.json",
    "experiment/e-CARE/增强后/bert+CI+Con.json",
    "experiment/e-CARE/增强前/bartold.json",
    "experiment/e-CARE/增强前/TwoOrevaluation_results.json"
]
model_names = ['ECCT(Bart)', 'ECCT(Bert)', 'Bart', 'Bert']

# 画混淆矩阵
plot_confusion_matrix(model_paths, model_names)
