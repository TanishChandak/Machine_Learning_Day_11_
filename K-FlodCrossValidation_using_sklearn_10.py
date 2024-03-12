from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()

# training and testing the dataset:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# Logistic Regression:
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("The accuracy of logistic regression: ",lr.score(X_test, y_test))

# SVC:
sv = SVC()
sv.fit(X_train, y_train)
print("The accuracy of SVC is: ",sv.score(X_test, y_test))

# Random forest classifier:
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("The accuracy of Random Forest Classifier is: ",rf.score(X_test, y_test))

# K_FoldCrossValidation :
from sklearn.model_selection import KFold
kf = KFold(n_splits=3) # n_splits = no of splits

for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

# Creating an function of checking the different models:
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    print("The accuracy of the model is:",model.score(X_test, y_test))

# same for the above model but here we use the function to simpliy it:
get_score(SVC(), X_train, X_test, y_train, y_test) # for SVC model
get_score(LogisticRegression(), X_train, X_test, y_train, y_test) # for Logistic model
get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test) # for Random forest model

# Same as the KFoldCrossValidation called StratifiedKFold:
from sklearn.model_selection import StratifiedKFold
fold = StratifiedKFold(n_splits=3) # this n_split is used to split the row matrix into number off

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in kf.split(digits.data):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]

    scores_logistic.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test))

print(scores_logistic)
print(scores_svm)
print(scores_rf)

# this CROSS_VAL_SCORE is generally doing the above for loop works:
from sklearn.model_selection import cross_val_score
cross_values_logistic = cross_val_score(LogisticRegression(), digits.data, digits.target)
print(cross_values_logistic)

cross_values_svc = cross_val_score(SVC(), digits.data, digits.target)
print(cross_values_svc)

cross_values_rf = cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target)
print(cross_values_rf)