# Arabic-Comments-Classification
## Readme: Arabic Comments Classification Project

This document provides an overview of the Jupyter Notebook titled `Arabic Comments classification.ipynb`, which implements a machine learning approach for classifying Arabic comments/reviews into sentiment categories.

---

### 🌟 Project Goal

The primary goal of this project is to classify Arabic reviews into three sentiment categories: **Negative (-1)**, **Neutral (0)**, and **Positive (1)**.

### 🛠️ Methodology

The project utilizes a pre-trained Arabic Sentence Transformer model to convert the text data into dense numerical vectors (embeddings), which are then used to train and evaluate two different classification models: Logistic Regression and XGBoost.

### 🚀 Workflow Summary

1.  **Setup and Data Loading:** Import necessary libraries (pandas, numpy, sklearn, torch, sentence_transformers) and load the dataset from the `train (1).xlsx` file.
2.  **Data Inspection:** Examine the first few rows and confirm the unique values in the `rating` column are -1, 0, and 1.
3.  **Preprocessing:** Map the original target ratings $(-1, 0, 1)$ to a zero-indexed scheme $(0, 1, 2)$ for model training.
4.  **Text Embedding:** Use a pre-trained Arabic Sentence Transformer model (`Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet`) to convert the review text into sentence embeddings, leveraging a GPU (`cuda`) if available.
5.  **Data Splitting:** Split the embedded data (`X`) and the modified labels (`Y`) into training and testing sets with a test size of 0.2 (20%).
6.  **Model Training and Evaluation:**
    * Train an **XGBoost Classifier** (`xgb.XGBClassifier`) with parameters `n_estimators=100`, `learning_rate=0.1`, and `max_depth=5`.
    * Train a **Logistic Regression** model (`LogisticRegression`) with `max_iter=500`.
    * Evaluate both models using the `classification_report` on the test set.

### 📊 Results

The classification report for the two models provides performance metrics across the three classes:

| Metric | Class 0 (Negative) | Class 1 (Neutral) | Class 2 (Positive) | Accuracy | Macro Avg | Weighted Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Classifier** | | | | | | |
| Precision | 0.77 | 0.29 | 0.83 | 0.81 | 0.63 | 0.78 |
| Recall | 0.77 | 0.01 | 0.89 | | 0.56 | 0.81 |
| F1-Score | 0.77 | 0.01 | 0.86 | | 0.55 | 0.79 |
| **Logistic Regression** | | | | | | |
| Precision | 0.77 | 0.25 | 0.83 | 0.81 | 0.62 | 0.78 |
| Recall | 0.76 | 0.02 | 0.90 | | 0.56 | 0.81 |
| F1-Score | 0.77 | 0.03 | 0.86 | | 0.55 | 0.79 |

**Observations:**

* **Overall Accuracy:** Both models achieve a similar overall accuracy of 0.81.
* **Class Imbalance:** There is a significant performance issue for **Class 1 (Neutral)** reviews, indicated by extremely low Recall (0.01 - 0.02) and F1-Score (0.01 - 0.03). This suggests the model is unable to correctly identify neutral reviews, likely due to a severe class imbalance problem.
* **Positive and Negative Classes:** Both models perform well on the more abundant **Positive (Class 2)** and **Negative (Class 0)** classes.

### 💡 Next Steps and Improvements

1.  **Address Class Imbalance:** The most critical next step is to employ techniques to handle the severe imbalance of the Neutral class:
    * Oversampling (e.g., SMOTE) or Undersampling.
    * Using class weights in the training process.
2.  **Hyperparameter Tuning:** Systematically tune the hyperparameters for the Logistic Regression and XGBoost models (e.g., regularization strength, learning rate, tree depth).
3.  **Explore Other Models:** Evaluate other classifiers like Support Vector Machines (SVM) or Neural Networks.
4.  **Model Optimization:** Explore alternative embedding models or fine-tune the existing Arabic Sentence Transformer on the specific task data.
