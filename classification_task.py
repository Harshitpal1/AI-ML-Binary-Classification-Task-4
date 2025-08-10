# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, classification_report

# 2. Load the Dataset [cite: 8]
cancer = load_breast_cancer()
data = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
data['target'] = cancer.target

# data.head()
# data.info()
# print("Class distribution:\n", data['target'].value_counts())

# 3. Define Features (X) and Target (y)
X = data.drop('target', axis=1)
y = data['target']

# 4. Train/Test Split and Feature Standardization [cite: 9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Fit the Logistic Regression Model [cite: 10]
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 6. Make Predictions
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# 7. Evaluate the Model [cite: 11]

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("--- Model Evaluation ---")
print("Confusion Matrix:\n", conf_matrix)

# Precision and Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Malignant (0)', 'Benign (1)']))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")

# Plot ROC-AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# 8. Threshold Tuning [cite: 12]
print("\n--- Threshold Tuning ---")
print("Default threshold is 0.5")

# Example with a new threshold of 0.7
custom_threshold = 0.7
y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

print(f"\nMetrics with threshold = {custom_threshold}:")
print(f"Precision: {precision_score(y_test, y_pred_custom):.4f} (Higher precision, fewer false positives)")
print(f"Recall: {recall_score(y_test, y_pred_custom):.4f} (Lower recall, more false negatives)")


# 9. Sigmoid Function Explanation & Visualization [cite: 12]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sigma = sigmoid(z)

plt.figure(figsize=(8, 4))
plt.plot(z, sigma)
plt.title("Sigmoid Function")
plt.xlabel("z (Linear combination of inputs)")
plt.ylabel("Sigmoid(z) (Probability)")
plt.grid(True)
plt.show()
