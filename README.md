# Logistic Regression Breast Cancer Classification

This repository contains my complete solution to **Task 4: Classification with Logistic Regression** for the Elevate AI & ML Internship. The task required building a binary classifier to distinguish between malignant and benign tumors using logistic regression.

---

## What I Did

### 1. **Loaded and Cleaned Data**
- Imported the Breast Cancer Wisconsin dataset (`data.csv`).
- Dropped irrelevant columns (such as `id`).
- Converted the target variable (`diagnosis`) from categorical (`M` for malignant, `B` for benign) to numerical values (`1` and `0`).

### 2. **Handled Missing Values**
- Used **SimpleImputer** to replace missing values in the dataset features with the mean of each column, ensuring compatibility with scikit-learn's `LogisticRegression`.

### 3. **Split Features and Target**
- Separated the dataset into input features (`X`) and target (`y`).
- Performed a **stratified train-test split** to maintain class proportions in both sets.

### 4. **Standardized Features**
- Scaled input features using **StandardScaler** for optimal model performance.

### 5. **Built and Trained the Model**
- Initialized a logistic regression model and trained it on the prepared data.

### 6. **Evaluated the Model**
- Made predictions and generated probabilities on the test set.
- Examined the **confusion matrix**, **classification report** (including precision, recall, and F1-score), and overall accuracy.

### 7. **Plotted the ROC Curve**
- Calculated and visualized the ROC curve to assess how well the model distinguishes between classes.
- Computed the **AUC** (Area Under Curve) for further insight into classification performance.

### 8. **Threshold Tuning**
- Identified and printed the optimal decision threshold (Youden’s J statistic) to potentially improve classification decisions.

### 9. **Visualized the Sigmoid Function**
- Plotted the sigmoid function to illustrate how logistic regression transforms linear predictions into probabilities.

---

## Results

- Achieved very high performance: **Accuracy, Precision, Recall, and F1-score all at 0.99** in the best model run.
- Model demonstrated strong ability to distinguish malignant from benign cases with minimal misclassifications.

---

## Getting Started

1. Clone this repo and ensure `data.csv` is available in the folder.
2. Install required libraries:
    ```
    pip install pandas numpy matplotlib scikit-learn
    ```
3. Run the script using:
    ```
    python task_4_logistic_regression.py
    ```

---

## Key Takeaways

- Logistic Regression is powerful for binary classification problems.
- Data preprocessing (handling missing values, scaling) is essential for reliable results.
- Evaluation with multiple metrics and threshold tuning helps in optimizing model performance.
- Visualizing model components (such as the sigmoid curve and ROC) aids understanding and communication.

---
