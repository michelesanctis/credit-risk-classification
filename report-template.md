# Module 12 Report Template

## Overview of the Analysis

​Credit risk arises when a borrower fails to return the asset or fulfill the loan obligation, leading to potential financial loss for the lender. 
In this analysis, we will utilize machine learning to examine a dataset of historical lending activities obtained from a peer-to-peer lending services company. The objective is to develop a model that can effectively assess the creditworthiness of borrowers based on historical lending data.  

I will employ a machine learning model to classify loans as healthy (low-risk) or non-healthy (high-risk) by analyzing the loan status data supplied by the lending company. 
For this task, the Logistic Regression Algorithm stands out as the optimal choice for our machine learning model due to its prevalence in predicting the likelihood of an outcome in binary classification challenges.

When applying the `value_counts` I noticed that the data was highly unbalanced when we compared healthy loans [0] with non-healthy loans [1], showing that we had 30 times more health loans.

According to the confusion matrix:
Out of the 18,765 loan status's that are healthy (low-risk), the model predicted 18,658 as healthy correctly and 107 as healthy incorrectly.
Out of the 619 loan status's that are non-healthy (high-risk), the model predicted 582 as non-healthy correctly and 37 as non-healthy incorrectly.

To generate a higher accuracy score and have the model catch more mistakes when classifying non-healthy loans, we can oversample the data using the `RandomOverSampler` module from the imbalanced-learn library, which adds more copies of the minority class (non-healthy loans) to obtain a balanced dataset.

Using the dataset provided by the lending company, I created a Logistic Regression Model fit with the oversampled data that generated an accuracy score of 99%, which turns out to be higher than the model fitted with imbalanced data. The oversampled model performs better due to the dataset being balanced. The models non-healthy loans recall value increased from 0.94 to 0.99 indicating that the model does an exceptional job in catching mistakes such as labeling non-healthy (high-risk) loans as healthy (low-risk).

According to the new confusion matrix:
Out of the 18,765 loan status's that are healthy, the model predicted 18,646 as healthy correctly and 119 as healthy incorrectly.
Out of the 619 loan status's that are non-healthy (high-risk), the model predicted 615 as non-healthy correctly and 4 as non-healthy incorrectly.


## Results

### Logistic Regression Model fitted with Imbalanced Data

1. The Logistic Regression model fitted with the Imbalanced DataSet predicted healthy loans 100% of the time and predicted non-healthy loans 84% of the time.

2. The model fitted with imbalanced data has a higher possibility of making these mistakes:
  - a healthy loan (low-risk) is classified as a non-healthy loan (high-risk).
  - a non-healthy loan (high-risk) is classified as a healthy loan (low-risk).

3. According to the models recall scores, the model made 1% of mistakes when predicting healthy loans and made 6% of mistakes when predicted non-healthy loans.

4. The model generated an accuracy score of 97% but could be improved due to the dataset being imbalanced.



### Logistic Regression Model fitted with Balanced (oversampled) Data

1. The Logistic Regression model fitted with the OverSampled DataSet predicted healthy loans 100% of the time and predicted non-healthy loans 84% of the time.


2. The model fitted with balanced (oversampled) data has a lower possibility of making these mistakes:
  - a healthy loan (low-risk) is classified as a non-healthy loan (high-risk).
  - a non-healthy loan (high-risk) is classified as a healthy loan (low-risk).

3. According to the models recall scores, the model made 1% of mistakes when predicting healthy loans and made 1% of mistakes when predicted non-healthy loans.

4. The model generated an accuracy score of 99% due to the dataset being balanced.


## Summary

A lending company might want a model that requires classifying healthy loans and non-healthy loans correctly most of the time.

Healthy loans being identified as a non-healthy loan might be more costly for a lending company since it might cause the loss of customers.
Non-healthy loans being identified as a healthy loan might also be more costly for a lending company due to the loss of funds being provided by the lender.

The Logistic Regression model fitted with OverSampled data performed much better than the model fitted with Imbalanced data due to the data being balanced and generating a higher accuracy score and a higher recall, indicating that the model will make extremely fewer mistakes when classifying non-healthy loans.

The lending company would most likely want fewer False Positives due to the high possibility of a lender loosing provided funds when classifying non-healthy loans as healthy. The data below is shown in the confusion matrices which indicates how many healthy/non-healthy loans the model predicted correctly/incorrectly.

Model fitted with Imbalanced Data:
  - 37 (FALSE POSITIVES) --> The actual value is healthy and the predicted value is non-healthy
  - 107 (FALSE NEGATIVES) --> The actual value is non-healthy and the predicted value is healthy

Model fitted with Balanced Data:
  - 4 (FALSE POSITIVES) --> The actual value is healthy and the predicted value is non-healthy
  - 119 (FALSE NEGATIVES) --> The actual value is non-healthy and the predicted value is healthy

According to the confusion matrices, the number of False Postives drastically decreases indicating the model will classify healthy & non-healthy loans correctly. Based off of this analysis, I would recommend using Model 2: Logistic Regression Model fitted with Balanced (oversampled) data.
