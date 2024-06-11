import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_auc_curves(model_paths, model_names):
    plt.figure(figsize=(12, 8))

    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        with open(model_path, 'r') as file:
            data = json.load(file)

        true_labels = np.array([entry["true"] for entry in data])
        predictions = np.array([entry["predictions"] for entry in data])

        # 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)

        # 计算PR曲线和AUC-PR
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        pr_auc = average_precision_score(true_labels, predictions)

        # 绘制ROC曲线
        plt.plot(fpr, tpr, label=f'{model_name} (AUC-ROC = {roc_auc:.2f})', linestyle='-')

    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        with open(model_path, 'r') as file:
            data = json.load(file)

        true_labels = np.array([entry["true"] for entry in data])
        predictions = np.array([entry["predictions"] for entry in data])

        # 计算PR曲线和AUC-PR
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        pr_auc = average_precision_score(true_labels, predictions)

        # 绘制PR曲线
        plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {pr_auc:.2f})', linestyle='--')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 绘制随机猜测线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR) / Recall')
    plt.ylabel('True Positive Rate (TPR) / Precision')
    plt.title('Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves')
    plt.legend(loc="lower right")
    plt.show()


# 文件路径和模型名称
model_paths = [
    "experiment/QA/增强后/bart+CI+Con.json",
    "experiment/QA/增强后/bert+CI+Con.json",
    "experiment/QA/增强前/bartold.json",
    "experiment/QA/增强前/Original_results.json"
]
model_names = ['ECCT(Bart)', 'ECCT(Bert)', 'Bart', 'Bert']

# 绘制AUC-ROC和AUC-PR曲线
plot_auc_curves(model_paths, model_names)
