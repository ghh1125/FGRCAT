import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Load data from JSON file
file_path = "experiment/e-CARE/增强后/bert+Con.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract predictions and true labels
predictions = [entry["predictions"] for entry in data]
true_labels = [entry["true"] for entry in data]

# Calculate standard deviation
std_dev = np.std(predictions)

# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

# Round the results to two decimal places

precision = round(precision, 3)
recall = round(recall, 3)
f1 = round(f1, 3)

# Print the results

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
