import json
import numpy as np

# Load data from JSON file
file_path = "experiment/QA/增强后/bert+Con.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract predictions and true labels
predictions = [entry["predictions"] for entry in data]
true_labels = [entry["true"] for entry in data]

# Calculate TP, FP, TN, FN for each class
class_counters = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
for pred, true in zip(predictions, true_labels):
    if pred == true:
        class_counters['TP'] += 1
    else:
        class_counters['FN' if true == 1 else 'FP'] += 1

# Calculate precision, recall, and F1 score
precision = class_counters['TP'] / (class_counters['TP'] + class_counters['FP'])
recall = class_counters['TP'] / (class_counters['TP'] + class_counters['FN'])
f1 = 2 * precision * recall / (precision + recall)

# Print the results
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))
