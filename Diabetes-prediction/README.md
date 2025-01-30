👨‍💻 Author
Shokoufeh Vakili– Machine Learning Enthusiast


# Machine-Learning-Projects
📌 Pima Indians Diabetes Prediction
A Machine Learning project analyzing the Pima Indians Diabetes dataset using K-Nearest Neighbors (KNN) and Logistic Regression.
The goal is to predict diabetes based on patient medical data.

🔍 Project Overview
Dataset: Pima Indians Diabetes Database
Goal: Predict whether a patient has diabetes based on diagnostic measurements.
Tech Stack: Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn
Models Used:
    K-Nearest Neighbors (KNN) ✅
    Logistic Regression ✅ 


## 📂 Dataset Description

| Feature                     | Description |
|-----------------------------|------------|
| `Pregnancies`               | Number of times pregnant |
| `Glucose`                   | Plasma glucose concentration |
| `BloodPressure`             | Diastolic blood pressure (mm Hg) |
| `SkinThickness`             | Triceps skin fold thickness (mm) |
| `Insulin`                   | 2-Hour serum insulin (mu U/ml) |
| `BMI`                       | Body mass index (weight in kg/(height in m)^2) |
| `DiabetesPedigreeFunction`  | Diabetes likelihood based on family history |
| `Age`                       | Age of the patient |
| `Outcome`                   | 0 = No Diabetes, 1 = Diabetes |



📊 Exploratory Data Analysis
✔ Checked for missing values and duplicates
✔ Standardized feature scaling using StandardScaler
✔ Visualized distributions, correlations, and outliers


🛠️ Model Training and Evaluation
🔹 K-Nearest Neighbors (KNN)
    Used hyperparameter tuning to find the best k value.
    Best KNN model achieved ~78% accuracy.
🔹 Logistic Regression
    Implemented Logistic Regression for comparison.
    Model performance evaluated using Confusion Matrix & Classification Report.


📌 Key Findings
📌 Feature importance analysis shows Glucose and BMI are highly correlated with diabetes.
📌 Logistic Regression performs similarly to KNN but is faster and interpretable.
📌 Model accuracy can be improved further with feature engineering and ensemble models.

