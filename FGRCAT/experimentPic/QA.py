import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

def plot_auc_curves(model_paths, model_names):
    plt.figure(figsize=(12, 8))

    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        with open(model_path, 'r') as file:
            data = json.load(file)

        true_labels = np.array([entry["true"] for entry in data])
        predictions = np.array([entry["predictions"] for entry in data])

        # 确保 predictions 是二维数组
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # Binarize the true labels for multi-class
        classes = np.unique(true_labels)
        true_labels_bin = label_binarize(true_labels, classes=classes)
        n_classes = true_labels_bin.shape[1]

        # Use OneVsRestClassifier for multi-class ROC and PR curves
        classifier = OneVsRestClassifier(LogisticRegression())
        classifier.fit(predictions, true_labels_bin)
        y_score = classifier.decision_function(predictions)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(n_classes):
            fpr[j], tpr[j], _ = roc_curve(true_labels_bin[:, j], y_score[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve for each class
        for j in range(n_classes):
            plt.plot(fpr[j], tpr[j], linestyle='-', label=f'{model_name} class {classes[j]} (AUC-ROC = {roc_auc[j]:.2f})')

        # Compute precision-recall curve and PR area for each class
        precision = dict()
        recall = dict()
        pr_auc = dict()
        for j in range(n_classes):
            precision[j], recall[j], _ = precision_recall_curve(true_labels_bin[:, j], y_score[:, j])
            pr_auc[j] = average_precision_score(true_labels_bin[:, j], y_score[:, j])

        # Compute micro-average precision-recall curve and PR area
        precision["micro"], recall["micro"], _ = precision_recall_curve(true_labels_bin.ravel(), y_score.ravel())
        pr_auc["micro"] = average_precision_score(true_labels_bin, y_score, average="micro")

        # Plot PR curve for each class
        for j in range(n_classes):
            plt.plot(recall[j], precision[j], linestyle='--', label=f'{model_name} class {classes[j]} (AUC-PR = {pr_auc[j]:.2f})')

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
    "experiment/QA/增强后/bert+CI+Con.json",
    "experiment/QA/增强前/Original_results.json"
]
model_names = ['FGRCAT(Bert)', 'Bert']

# 绘制AUC-ROC和AUC-PR曲线
plot_auc_curves(model_paths, model_names)
