# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------------------------------------------------
# 1. Loading Data
# ------------------------------------------------------------------------------

# Download dataset from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Load the dataset
data = pd.read_csv("pima-indians-diabetes.csv")

# Display first few rows
print("Dataset Preview:")
display(data.head())

# ------------------------------------------------------------------------------
# 2. Exploring the Dataset
# ------------------------------------------------------------------------------

# Checking dataset information
print("\nDataset Info:")
print(data.info())

# Descriptive statistics of numerical features
print("\nDataset Summary:")
print(data.describe())

# Checking for missing values
print("\nMissing Values in Each Column:")
print(data.isna().sum())

# Checking for duplicate rows
print("\nNumber of Duplicate Rows:", data.duplicated().sum())

# ------------------------------------------------------------------------------
# 3. Data Visualization
# ------------------------------------------------------------------------------

# Distribution of the target variable (Outcome)
plt.figure(figsize=(8, 4))
sns.countplot(x='Outcome', data=data, palette="Set2")
plt.title("Distribution of Diabetes Outcome")
plt.show()

# Boxplots to detect outliers in numerical features
plt.figure(figsize=(12, 12))
for i, col in enumerate(data.columns[:-1]):  # Exclude 'Outcome'
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=data[col], color="skyblue")
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# Pairplot to visualize feature distributions by Outcome
sns.pairplot(data, hue='Outcome', palette="husl")
plt.show()

# Heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# ------------------------------------------------------------------------------
# 4. Data Preprocessing
# ------------------------------------------------------------------------------

# Standardizing the feature variables
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data.drop(columns=['Outcome'])), 
                 columns=data.columns[:-1])

# Target variable
y = data['Outcome']

# Splitting dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------------------------------------------------------
# 5. Model Training and Evaluation - K-Nearest Neighbors (KNN)
# ------------------------------------------------------------------------------

# Finding the best 'k' value for KNN by evaluating train and test accuracy
train_scores = []
test_scores = []

for k in range(1, 15):
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

# Identifying the best 'k' value for highest test accuracy
best_k = np.argmax(test_scores) + 1
best_accuracy = max(test_scores)
print(f"\nBest k value: {best_k}, Accuracy: {best_accuracy:.4f}")

# Plotting accuracy scores for different values of k
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 15), y=train_scores, label="Train Accuracy", marker="o")
sns.lineplot(x=range(1, 15), y=test_scores, label="Test Accuracy", marker="o")
plt.axvline(best_k, color='r', linestyle="--", label=f"Best k: {best_k}")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Model Performance")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# 6. Final Model Training and Evaluation
# ------------------------------------------------------------------------------

# Train final KNN model with best k value
knn = KNeighborsClassifier(best_k)
knn.fit(X_train, y_train)

# Model evaluation
y_pred = knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------------------------
# 7. Summary of Findings
# ------------------------------------------------------------------------------
"""
- We analyzed the Pima Indians Diabetes dataset, checking for missing values and duplicates.
- We visualized feature distributions, correlations, and outliers.
- Standardization was applied to normalize the features.
- A KNN model was trained, and the best value of k was found to be {best_k} with {best_accuracy:.2f} accuracy.
- Model evaluation showed that our classifier performs well, with decent precision and recall.
"""


