[![Badge](https://img.shields.io/badge/Author-Fabiano_Manetti-%237159c1?style=flat-square&logo=ghost)](https://github.com/FabianoManetti/) [![Linkedin Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fabiano-manetti/)

# Dimensionality Reduction - Customer Satisfaction Prediction

<div align="center">
<img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=yellow"> </img>
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"></img>
</div>


**This is part of the second training course of https://www.datascienceacademy.com.br/ Data Scientist program.**

<center><img src="customer_satisfaction.jpg"></center><br>

## Problem Context

Santander Bank was asking Data Scientists to help them identify **dissatisfied** customers at the beggining of the relationship. This would allow Santander to create proactive measures in order to retain that client.

For this project, it was requested an **accuracy of, at least, 70%**.

The dataset consisted of many anonymous features from different customers that demanded the use of **dimensionality reduction** tools.

## Feature Reduction

### First reduction: Variance Threshold

The **variance** is simply the average of the squared differences from the mean. It is import in the context of machine learning because features with 0 or close to 0 variance are generally **useless** in terms of prediction power.

For the purpose of this project, we made use of the `VarianceThreshold` estimator. It selected all the features in our dataset whose variance is higher than a set threshold.

* After it, we were able to **eliminate** a great number of columns, 325 to be exact. This result showed us that many features in our initial dataframe were potentially useless to be used as predictors.

### Second reduction: Multicollinearity

Multicollinearity refers to the occurance of high **intercorrelations** among two or more independent variables. 

Multicollinearity causes problems in the interpretability of the model result, for this reason it is advisable to drop features that present high correlation with each other.

* Once again, we were able to **reduce** the dimensionality of the dataframe.

### Third reduction: Recursive Feature Elimination

Recursive Feature Elimination, or RFE for short, is a feature reduction method that works by **recursively training** diferent sets of features and ranking them by their importance with the use of an estimator.

The supervised learning estimator needs to provide information about feature importance. For the purspose of this project, we used `Logistic Regression` as the auxiliary model.

<center><img src="images/rfecv.png"></center><br>

* RFECV model achieved the highest accuracy model with **11 features**.

### Fourth reduction: Random Forest Feature Importance

The ensemble estimator **Random Forest** has the property `Feature Importance`, which returns us the features that contribute most to the reduction of the Gini impurity criterion.

We made used again of the `Pipeline`, `RobustScaler` and  `SMOTE` methods and test the **accuracy** of the model with cross validation.

<center><img src="images/feature_importance.png"></center><br>

* It is interesting to note that only a few features in our dataframe were responsible for most of the power of prediction.

## Testing predictive models

At this part of the project, we tested different algorithms using the final set of features that we found in our fourth reduction step. Our goal was to improve or at least keep our current accuracy score.

