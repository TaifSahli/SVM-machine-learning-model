
# SVM-Based Purchase Prediction Model

## Project Overview

In this project, we designed, implemented, and evaluated a Support Vector Machine (SVM) model to predict whether a user will make a purchase based on their demographic and financial data. This project aims to provide valuable insights for targeted marketing strategies.

## Table of Contents

1. [Project Goals](#project-goals)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Objective](#objective)
5. [Approach](#approach)
6. [Model Implementation](#model-implementation)
7. [Evaluation and Results](#evaluation-and-results)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [Team Members](#team-members)
11. [References](#references)

## Project Goals

The primary goal of this project is to develop a robust SVM model that can accurately classify whether a user will purchase a particular product based on their demographic and financial features.

## Problem Statement

Predict whether a user will make a purchase or not based on features such as gender, age, and estimated salary.

## Dataset

The dataset contains information about users, including:
- **User ID:** A unique identifier for each user (not used in the model)
- **Age:** The age of the user
- **Gender:** The gender of the user (encoded as 0 for male, 1 for female)
- **Estimated Salary:** The estimated annual salary of the user
- **Purchased:** Indicates whether the user has purchased the product (1 for purchased, 0 for not purchased)

## Objective

Build a machine learning model using Support Vector Machines (SVM) to classify users as potential purchasers or non-purchasers based on their gender, age, and estimated salary.

## Approach

1. **Data Preprocessing:**
   - Dropped unnecessary columns (User ID).
   - Encoded the gender column into numerical values.
   - Applied feature scaling to standardize numerical features.

2. **Model Training:**
   - Split the dataset into training and testing sets using `train_test_split`.
   - Trained a linear SVM model on the training set using `SVC` from scikit-learn.

3. **Model Evaluation:**
   - Evaluated model performance on the test set using accuracy as the primary metric.

## Model Implementation

The model was implemented in Python using the following libraries:
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computations.
- **scikit-learn (sklearn):** For machine learning algorithms and tools.
- **Matplotlib:** For data visualization.

## Evaluation and Results

- **Accuracy:** The SVM model achieved an accuracy of 89.0% on the test set, indicating effective learning and generalization.
- **Predictions:** The model was able to predict purchase behavior based on user features.

### Example Predictions:
- **New_Instance1 (Purchased):** Female, 30 years old, estimated salary of 208754.
- **New_Instance2 (Not Purchased):** Male, 38 years old, estimated salary of 29871.

## Conclusion

This project demonstrated the potential of SVMs in predicting user purchase behavior based on demographic and financial features. The model provided valuable insights into the importance of age and estimated salary in predicting purchases.

## Future Work

- **Data Expansion:** Collect additional data to enhance the model's predictive capabilities.
- **Hyperparameter Tuning:** Explore advanced techniques like grid search to optimize model performance.
- **Feature Importance Analysis:** Conduct a thorough analysis to understand the significance of different user characteristics.
- **Real-World Application:** Deploy the trained model in real-world applications for real-time predictions and marketing strategy optimization.

## Team Members

- Taif Fadhel Alsahli
- Kindah Turki Alotaibi
- Latifah Ibrahim Alothman
- Taif Faris Alatif

## References

- [Classifying Data Using Support Vector Machines (SVMs) in Python](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/)
- [Introduction to Support Vector Machines (SVM)](https://www.geeksforgeeks.org/introduction-to-support-vector-machines-svm/)
- [Understanding Support Vector Machine (SVM) with Example](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)
- [scikit-learn SVC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
