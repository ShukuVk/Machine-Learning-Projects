# ------------------------------------------------------------------------------
# 7. Logistic Regression Model
# ------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Model evaluation
y_pred_lr = log_reg.predict(X_test)

# Model accuracy
accuracy_lr = log_reg.score(X_test, y_test)
print(f"\nLogistic Regression Accuracy: {accuracy_lr:.4f}")

# Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Classification report
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
