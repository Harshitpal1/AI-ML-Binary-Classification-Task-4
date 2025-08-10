# AI-ML-Binary-Classification-Task-4
# Task 4: Classification with Logistic Regression

This repository contains the solution for the "AI & ML INTERNSHIP" task focused on building a binary classifier.

## Objective

The primary objective is to build a binary classifier using logistic regression. This project demonstrates the complete workflow from data preparation to model evaluation and interpretation.

## Dataset

This project utilizes the **Breast Cancer Wisconsin Dataset**, a common dataset for binary classification tasks. The goal is to classify tumors as either malignant or benign.

## Tools Used

* Python
* Scikit-learn
* Pandas
* Matplotlib

## Methodology

The project followed the steps outlined in the task's mini-guide:

1.  **Dataset Selection:** A suitable binary classification dataset was chosen as recommended.
2.  **Data Preprocessing:** The data was split into training and testing sets, and the features were standardized for optimal model performance.
3.  **Model Training:** A Logistic Regression model was fitted to the preprocessed training data.
4.  **Model Evaluation:** The model's performance was evaluated using a confusion matrix, precision, recall, and the ROC-AUC score.
5.  **Threshold and Sigmoid Function:** The project includes an analysis of how tuning the decision threshold impacts outcomes and an explanation of the sigmoid function's role.

## How to Run the Code

1.  Clone the repository:
    ```bash
    git clone <your-repo-link>
    ```
2.  Navigate to the project directory:
    ```bash
    cd <repository-folder>
    ```
3.  Install the required libraries:
    ```bash
    pip install pandas scikit-learn matplotlib
    ```
4.  Run the Python script:
    ```bash
    python classification_task.py
    ```

## Interview Questions & Answers

This section addresses the interview questions provided in the task description.

#### 1. How does logistic regression differ from linear regression?
Linear regression predicts continuous values (e.g., price, temperature), while logistic regression predicts the probability of an instance belonging to a specific class, making it suitable for classification tasks. The output of logistic regression is transformed by a sigmoid function to be between 0 and 1.

#### 2. What is the sigmoid function?
The sigmoid function is a mathematical "S"-shaped curve that maps any real-valued number into a value between 0 and 1. In logistic regression, it's used to convert the linear combination of inputs into a probability score.

#### 3. What is precision vs recall?
* **Precision** measures the accuracy of positive predictions. It answers: "Of all the instances the model predicted as positive, how many were actually positive?" It is calculated as `True Positives / (True Positives + False Positives)`.
* **Recall** (or Sensitivity) measures the model's ability to find all the actual positive instances. It answers: "Of all the actual positive instances, how many did the model correctly identify?" It is calculated as `True Positives / (True Positives + False Negatives)`.

#### 4. What is the ROC-AUC curve?
The **ROC (Receiver Operating Characteristic) curve** is a graph showing the performance of a classification model at all classification thresholds. It plots the True Positive Rate (Recall) against the False Positive Rate. The **AUC (Area Under the Curve)** measures the entire two-dimensional area underneath the entire ROC curve, providing an aggregate measure of performance across all possible thresholds. A model with an AUC of 1.0 is perfect, while an AUC of 0.5 is no better than random guessing.

#### 5. What is the confusion matrix?
A confusion matrix is a table that summarizes the performance of a classification model. It shows the number of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN), giving a detailed breakdown of correct and incorrect predictions for each class.

#### 6. What happens if classes are imbalanced?
If classes are imbalanced, a model can achieve high accuracy by simply predicting the majority class. This leads to poor performance in identifying the minority class, which is often the class of interest. Metrics like accuracy become misleading, and techniques like resampling, using class weights, or focusing on metrics like Precision, Recall, and F1-score are necessary.

#### 7. How do you choose the threshold?
The classification threshold (default is 0.5) is chosen based on the business problem and the trade-off between precision and recall. If
