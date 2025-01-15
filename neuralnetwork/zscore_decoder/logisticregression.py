""" Created on Thu Nov 14 14:05:45 2024
    @author: dcupolillo """

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import flammkuchen as fl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report)
from spyne.neuralnetwork.zscore_decoder.zscoredataset import ZScoreDataset

# Load data
data = fl.load(r"neuralnetwork\zscore_decoder\zscore_labels.h5")
dataset = ZScoreDataset(data)

# Prepare features and labels
X = np.array([dataset[i][0].numpy() for i in range(len(dataset))])
y = np.array([dataset[i][1].item() for i in range(len(dataset))])

# Split data into training, validation, and test sets
validation_split = 0.3
test_split = 0.15

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=(validation_split + test_split),
    random_state=42,
    stratify=y)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp,
    test_size=(test_split / (validation_split + test_split)),
    random_state=42,
    stratify=y_temp)

# Hyperparameter tuning using Grid Search
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(
    LogisticRegression(class_weight='balanced', random_state=42),
    param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_['C'])

# Train logistic regression model with the best parameter
lr_model = LogisticRegression(
    C=grid_search.best_params_['C'], class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

fl.save(r"neuralnetwork\zscore_decoder\lr_model.h5",
        {"lr_model": lr_model})


# Evaluate model on validation set
y_valid_pred = lr_model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print(f"Validation Accuracy: {valid_accuracy:.2f}")

# Evaluate model on test set
y_test_pred = lr_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)

print(classification_report(
    y_test, y_test_pred, target_names=["No Event", "Event"]))

# Normalize confusion matrix to percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot Confusion Matrix with percentages
plt.figure(figsize=(8, 6))

annotations = np.array([
    [f"{cm_percentage[i, j]:.2f}%\n({cm[i, j]})" for j in range(cm.shape[1])]
    for i in range(cm.shape[0])
])

sns.heatmap(
    cm_percentage,
    annot=annotations, fmt='', cmap='Blues',
    xticklabels=["No Event", "Event"],
    yticklabels=["No Event", "Event"],
    cbar_kws={'label': 'Percentage (%)'})

plt.xlabel('Predicted Label')
plt.ylabel('True Label')