import re

# Open the file and read its contents
file_path = "experiment/QA/增强后/bert+CI+Con.txt"
with open(file_path, 'r') as file:
    content = file.read()

# Use regular expression to find all occurrences of "Dev Accuracy"
dev_accuracy_matches = re.findall(r'Dev Accuracy:\s+(\d+\.\d+)', content)

# Convert the matched strings to floating-point numbers
dev_accuracy_values = [float(match) for match in dev_accuracy_matches]

# Calculate maximum, minimum, and average
max_accuracy = max(dev_accuracy_values)
min_accuracy = min(dev_accuracy_values)
avg_accuracy = sum(dev_accuracy_values) / len(dev_accuracy_values)

# Print the results
print("Maximum Dev Accuracy:", max_accuracy)
print("Minimum Dev Accuracy:", min_accuracy)
print("Average Dev Accuracy:", avg_accuracy)
