# Regression model for california hausing


# Features Description
'''
MedInc: median income in block group
* HouseAge: median house age in block group
* AveRooms: average number of rooms per household
* AveBedrms: average number of bedrooms per household
* Population: block group population
* AveOccup: average number of household members
* Latitude: block group latitude
* Longitude: block group longitude
'''

# Import Libraries

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt


# Importing dataset as dictionary and print a description

housing = fetch_california_housing(as_frame=True)  # load data frame
print("The differnet keys of the data-frame are: ", housing.keys())  # print keys
print("The dimension of the data-frame is: ", housing.frame.shape)   # print shape
print("\nThe first 5 values of the data-frame are:")   
print(housing.frame.head(5)) # print firt 5 rows of data-frame


# Assign features values to numpy array X and target values to numpy array y

X = housing.data.to_numpy()   # Features
y = housing.target.to_numpy() # Target
print("feature dimension: dim(X) = ", X.shape)
print("target dimension: dim(y) = ", y.shape)


# Create histograms of features and target

plt.figure(figsize=(15, 14))
for j,name in enumerate(housing.feature_names):
    plt.subplot(3, 3, j+1)
    plt.hist(X[:,j], bins = "auto", label='feature')
    plt.title(f"Histogram of {name}")
    plt.legend()
    plt.grid(True, axis = "y")
    
plt.subplot(3, 3, len(housing.feature_names)+1)
plt.hist(y, bins = 10, label='target', color = "red")
plt.title(f"Histogram of MedHouseVal")
plt.legend()
plt.grid(True, axis = "y")
plt.tight_layout()
plt.savefig("distribution_of_features_target.png", dpi = 300)


# Scatter plots

plt.figure(figsize=(15,14))
for j,name in enumerate(housing.feature_names):
    plt.subplot(4,2,j+1)
    plt.title(f"Scatter target vs {name}")
    plt.scatter(X[:,j],y, s = 0.3)
    plt.xlabel(name)
    plt.ylabel("target")
    plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_target_feature.png", dpi = 300)


# Split the total dataset in a train and a test sample using scikit-learn

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)


# Print dimension of split train/test samples

print("dim(X_train) = ", X_train.shape)
print("dim(X_test) = ", X_test.shape)
print("dim(y_train) = ", y_train.shape)
print("dim(X_test) = ", y_test.shape)


# =================================Linear MODEL==============================================
# Fit a linear regression model using the training dataset 

model1 = LinearRegression()
model1.fit(X_train, y_train)
print("Fit of the model using linear regression gives the weights:\n")
print("Bias term w_0 = ", model1.intercept_)
for i in range(1,9):
    print(f"w_{i} = {model1.coef_[i-1]}" )


# Get the predicted model output using the training dataset

y_train_pred = model1.predict(X_train)


# Make a scatter plot of the true target value vs the predicted value 

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.title("Scatter plot train vs train_prediction")
plt.scatter(y_train, y_train_pred, s=1)
plt.plot(range(0,5),range(0,5), color = "black")
plt.xlabel("y_train")
plt.ylabel("y_train_pred")
plt.grid()

# Plot the difference (y_train - y_train_pred) in a histogram.

plt.subplot(1,2,2)
plt.title("Histogram y_train - y_train_pred")
plt.hist(y_train - y_train_pred)
plt.grid(axis = "y")

plt.savefig("prediction_vs_target_train.png", dpi = 300)


# Calculate the root mean square error (RMS) between `y_train` and `y_train_pred`. 

print('\nTrain sample: RMS = %.3f' 
      % (np.sqrt(mean_squared_error(y_train,y_train_pred))))  # use scikit-learn function mean_squared_error()


# Repeat prediction, scatter plot, histogram and RMS (test)

y_test_pred = model1.predict(X_test)   # prediction on test

# scatter (test)
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.title("Scatter plot test vs test_prediction")
plt.scatter(y_test, y_test_pred, s = 1, c = "red")
plt.plot(range(0,7),range(0,7), color = "black")
plt.xlabel("y_test")
plt.ylabel("y_test_pred")
plt.grid()

# hist (test)
plt.subplot(1,2,2)
plt.hist(y_test - y_test_pred, color = "red")
plt.title("Histogram y_test - y_test_pred")
plt.ylabel("difference")
plt.xlabel("frequency")
plt.grid(axis = "y")

plt.savefig("prediction_vs_target_test.png", dpi = 300)

# rms (test)
print("Root Mean Square Value = ", np.sqrt(mean_squared_error(y_test, y_test_pred)))


#=================================RIDGE MODEL=====================================================================
# define Ridge model (L2 norm) 

model2 = Ridge(alpha=1) # Alpha sets the lambda regularization parameter =1


# Train and predict for test

model2.fit(X_train,y_train)
y_test_pred_ridge = model2.predict(X_test)
print("Root Mean Square value for Ridge = ", np.sqrt(mean_squared_error(y_test, y_test_pred_ridge)))


# Compare different regularization terms

for i in range(1,11):
    model3 = Ridge(alpha = i)
    model3.fit(X_train,y_train)
    y_test_pred_lambdas= model3.predict(X_test)
    rms_pred_lambdas = np.sqrt(mean_squared_error(y_test, y_test_pred_lambdas)) 
    print(f"RMS for Ridge w/lambda = {i}: RMS = {rms_pred_lambdas:.4f}")


# Scatter (Ridge test)
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.title("Ridge Scatter y_test vs y_test_pred_ridge")
plt.scatter(y_test, y_test_pred_ridge, s = 1, c = "green")
plt.plot(range(0,7), range(0,7), "k")
plt.xlabel("y_test")
plt.ylabel("y_test_pred_ridge")
plt.grid()


# Hist (Ridge test) 
plt.subplot(1,2,2)
plt.title("Ridge y_test - y_test_pred_ridge")
plt.hist(y_test - y_test_pred_ridge, bins = 10, color = "green")
plt.grid(axis = "y")
plt.savefig("prediction_vs_target_test_ridge.png", dpi = 300)


# ============================Lasso Penalty Model==========================================
# Use Lasso penalty

model4 = Lasso(alpha=1)
model4.fit(X_train,y_train)
y_test_pred_lasso = model4.predict(X_test)

for i in range(1,11):
    model5 = Lasso(alpha = i)
    model5.fit(X_train,y_train)
    y_test_pred_Lasso = model5.predict(X_test)
    rms_pred_lasso = np.sqrt(mean_squared_error(y_test, y_test_pred_Lasso))
    print(f"RMS for Lasso w/lambda = {i} = {rms_pred_lasso:.3f}")


# Scatter (Lasso test)
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.title("Lasso Scatter y_test vs y_test_pred_lasso")
plt.scatter(y_test, y_test_pred_lasso, s = 1, c = "purple")
plt.plot(range(0,7), range(0,7), "k")
plt.xlabel("y_test")
plt.ylabel("y_test_pred_lasso")
plt.grid()


# Hist  (Lasso test)
plt.subplot(1,2,2)
plt.title("Lasso y_test - y_test_pred_lasso")
plt.hist(y_test - y_test_pred_lasso, bins = 10, color = "purple")
plt.grid(axis = "y")

plt.savefig("prediction_vs_target_test_lasso.png", dpi = 300)


# ========================  Estimating model performance: Cross-validation  =========================================================================================

# Apply the cross-validation to the other models 

from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=False)


# calculating score

scores_Linear = cross_val_score(model1,X, y, scoring='neg_mean_squared_error', cv=cv)
scores_Lasso = cross_val_score(model2,X, y, scoring='neg_mean_squared_error', cv=cv)
scores_Ridge = cross_val_score(model4,X, y, scoring='neg_mean_squared_error', cv=cv)


# absolute values

scores_Linear = np.absolute(scores_Linear)
scores_Lasso = np.absolute(scores_Lasso)
scores_Ridge = np.absolute(scores_Ridge)

# mean of score and erro

print('Mean RMS for Linear: %.2f +- %.2f' % (np.mean(np.sqrt(scores_Linear)),np.std(np.sqrt(scores_Linear))))
print('Mean RMS for Lasso: %.2f +- %.2f' % (np.mean(np.sqrt(scores_Lasso)),np.std(np.sqrt(scores_Lasso))))
print('Mean RMS for Ridge: %.2f +- %.2f' % (np.mean(np.sqrt(scores_Ridge)),np.std(np.sqrt(scores_Ridge))))