# Golden-Projects_Breast-Cancer-Classification
Develop an accurate model to distinguish between malignant and benign tumors, supporting early breast cancer detection. Prioritize high accuracy, sensitivity for early detection, and feature importance analysis. Aim for model generalization, interpretability, and ethical considerations to ensure responsible and unbiased usage in the medical field.

The dataset employed for this project is derived from Kaggle - https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data . It encompasses details such as:

Tumor Classification: Instances labeled to indicate whether tumors are malignant (M) or benign (B).
Diagnostic Features: A set of features describing tumor characteristics, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension, and more.
Biopsy Timestamps: Timestamps for diagnostic assessments, offering temporal context for data analysis.
Patient-Related Information: Patient-specific details such as age, menopausal status, and individual cell features.

## Data Splitting
The dataset is partitioned into training and testing sets, ensuring a comprehensive evaluation of the classification model's performance.

## Data Cleaning
Rigorous data cleaning procedures, including handling missing values and addressing outliers, are implemented to prepare a reliable dataset for analysis.

## Exploratory Data Analysis
In-depth EDA techniques are employed to extract insights into the characteristics of malignant and benign tumors. Visualization tools uncover patterns, correlations, and anomalies, contributing to a deeper understanding of the data.

## Feature Engineering
Feature engineering strategies are applied to enhance the model's predictive performance. This involves creating new features, transforming existing ones, or extracting meaningful information to enrich the dataset.

## Feature Scaling
Certain machine learning models benefit from feature scaling. Techniques such as scaling and normalization are applied to ensure consistent and effective model training.

## Data Imbalance
Techniques to address potential class imbalance, such as the Synthetic Minority Oversampling Technique (SMOTE), are employed to enhance the representation of minority classes, particularly relevant for malignant tumors.

## Preprocessing Function
A dedicated Python function, cancer_data_prep(dataframe), is crafted to streamline and execute all preprocessing steps on the test data. This function adeptly handles missing values by imputing them with the mean value derived from the training set.

## Models Training
State-of-the-art machine learning models are utilized for classification tasks. Thorough training and evaluation processes are conducted to select the most effective model, considering metrics like accuracy, precision, recall, and F1-score.


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'X' is the feature matrix and 'y' is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate the Support Vector Machine (SVM) classifier
svm_model = SVC(kernel='linear', random_state=42)

# Train the model on the scaled training data
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_result)
