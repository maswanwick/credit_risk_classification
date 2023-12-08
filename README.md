# Credit Risk Analysis Report

## Overview of the Analysis

The purpose of this exercise was to use supervised machine learning to predict an applicant's credit risk based on historic loans.  These previous loans used the following factors as input to determine the overall loan risk:
* Loan amount
* Interest rate
* Borrower income
* Debt-to-income ratio
* Number of accounts
* Derogatory marks
* Total debt

Approximately 97% of the 77,500 historic loans are successful, leaving 3% that are unsuccessful.  My job is to look at the above factors to see if I can determine indicators of a credit risk applicant.

To calculate the prediction, I seperated the loan status from the dataset, which is the indicator for the loan's success.  This is the 'y' label.

Then I grouped the above factors into the features.  This is the 'X' label.

Using train_test_split, I trained the LogisticRegression model with the X and y train data.

Once the model was trained, I performed a predicition on the test data and compared the accuracy using balanced_accuracy_score, confusion_matrix, and classification_report.

Finally, since the historical data had such a wide margin between successful and unsuccessful loans (97% to 3%), I used the RandomOverSample model to create an even distribution between successful and unsuccessful loan statuses to see if it impacted the results, using the same methods documented above.

## Results

* LogisticRegression model:
  * Of all the loans the model predicted as healthy, 100% of them actually were. (Precision 0 = 1.00)
  * Of all the loans the model predicted as high-risk, 85% of them actually were. (Precision 1 = 0.85)
  * Of all the loans that were actually healthy, the model predicted 99% of them. (Recall 0 = 0.99)
  * Of all the loans that were actually high-risk, the model predicted 91% of them. (Recall 1 = 0.91)
  * Both F1 scores are very close to 1 (0 = 1.0, 1 = 0.88), with an overall accuracy of 99%, therefore, the logistic regression model does a good job of predicting both healthy and high-risk loans.

* RandomOverSample model:
  * Of all the loans the model predicted as healthy, 91% of them actually were. (Precision 0 = 0.91)
  * Of all the loans the model predicted as high-risk, 99% of them actually were. (Precision 1 = 0.99)
  * Of all the loans that were actually healthy, the model predicted 100% of them. (Recall 0 = 1.00)
  * Of all the loans that were actually high-risk, the model predicted 90% of them. (Recall 1 = 0.90)
  * Both F1 scores are very close to 1 (0 = 0.95, 1 = 0.95) with an overall accuracy of 95%, therefore, the logistic regression model does a good job of predicting both healthy and high-risk loans.

## Summary

Both models have a high overall accuracy score (LogisticRegression = 99%, RandomOverSample = 95%), so both would be a good choice when predicting borrower credit risk.  However, from a business standpoint, even though the LogisticRegression model had a 99% overall accuracy, it only had an 85% accuracy when predicting high-risk loans that actaully were high-risk.  Which means, there is a chance that the lender would reject an applicant that would actually be a successful loan.  So the lender might be missing out on potential revenue with this model.  Conversely, the RandomOverSample model was able to accurately predict 99% of these high-risk loans.  But, the cost of this increased accuracy comes at the expense of the accuracy of healthy loans - which means they might end up loaning to someone that is actually a credit risk.  So from a risk perspective, the LogisticRegression model seems to be the better options, since missing out on potential revenue is a better option than lending to a borrower who ultimately defaults.

