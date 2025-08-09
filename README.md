HOA Activity Classification using Logistic Regression and SVM

Overview
This project demonstrates the implementation of Logistic Regression (from scratch) and Support Vector Machine (SVM) (using scikit-learn) to classify HOA (Home Owners Association) activities based on a dataset.

It covers:
Data preprocessing and encoding.
Exploratory data visualization.
Logistic Regression cost function & gradient descent.
Model training and evaluation.
Comparison with SVM classification.

The dataset used is:
pd_hoa_activities_cleaned.csv
Columns include:
task – Activity performed.
duration – Duration of task.
age – Age of participant.
class – Label indicating HOA or Non-HOA activity.

Features Implemented
1.) Data Preprocessing:
Encoded categorical task values into numerical indexes.
Encoded target class into binary labels (0 for HOA, 1 for Non-HOA).
Split into training (80%) and testing (20%) sets.
2.) Exploratory Data Analysis:
Histograms for age and duration.
Bar chart for task frequency.
3.) Logistic Regression (From Scratch):
Implemented Sigmoid function.
Implemented Cost function.
Gradient Descent parameter updates.
Tested with different initializations & learning rates.
Chose optimal parameters based on lowest cost.
4. Prediction & Accuracy:
Accuracy calculation for both training and testing sets.
5. Support Vector Machine (SVM):
Implemented using scikit-learn with a linear kernel.
Trained and evaluated on the same dataset for comparison.
