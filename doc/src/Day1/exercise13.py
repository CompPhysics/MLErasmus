import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
#from sklearn.linear_model import LinearRegression, Ridge, Lasso
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(3155)

x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

# number of features p (here degree of polynomial
p = 3
#  The design matrix now as function of a given polynomial
X = np.zeros((len(x),p))
X[:,0] = 1.0
X[:,1] = x
X[:,2] = x*x
# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# matrix inversion to find beta
OLSbeta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print(OLSbeta)
# and then make the prediction
ytildeOLS = X_train @ OLSbeta
print("Training R2 for OLS")
print(R2(y_train,ytildeOLS))
print("Training MSE for OLS")
print(MSE(y_train,ytildeOLS))
ypredictOLS = X_test @ OLSbeta
print("Test R2 for OLS")
print(R2(y_test,ypredictOLS))
print("Test MSE OLS")
print(MSE(y_test,ypredictOLS))

# Repeat now for Ridge regression and various values of the regularization parameter
I = np.eye(p,p)
# Decide which values of lambda to use
nlambdas = 100
MSEPredictLasso = np.zeros(nlambdas)
MSEPredictRidge = np.zeros(nlambdas)
lambdas = np.logspace(-4, 0, nlambdas)
for i in range(nlambdas):
    lmb = lambdas[i]
    # add ridge
    clf_ridge = skl.Ridge(alpha=lmb).fit(X_train, y_train)
    clf_lasso = skl.Lasso(alpha=lmb).fit(X_train, y_train)
    yridge = clf_ridge.predict(X_test)
    ylasso = clf_lasso.predict(X_test)
    MSEPredictLasso[i] = MSE(y_test,ylasso)
    MSEPredictRidge[i] = MSE(y_test,yridge)
#then plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSEPredictRidge, 'r--', label = 'MSE Ridge Test')
plt.plot(np.log10(lambdas), MSEPredictLasso, 'g--', label = 'MSE Lasso Test')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()





