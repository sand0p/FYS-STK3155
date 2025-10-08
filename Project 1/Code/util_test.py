from util import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as OLS_sklearn, Ridge as Ridge_sklearn

"""
Testing the implementation of OLS and Ridge regression against sklearn's implementation.
"""

# Generating data
np.random.seed(2025)

n_points = 50

noise = 0.01 * np.random.randn(n_points,1)
x = np.linspace(-1, 1, n_points).reshape(-1,1)
y = Runge(x) + noise

X = PolynomialFeatures(degree = 15).fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
y_offset = np.mean(y_test)


# OLS Test
def test_OLS():
	theta_OLS = OLS(X_train_s, y_train)
	predictions_OLS = X_test_s @ theta_OLS + y_offset


	predictions_OLS_sklearn = OLS_sklearn(fit_intercept=False).fit(X_train_s, y_train).predict(X_test_s) + y_offset

	assert np.allclose(predictions_OLS, predictions_OLS_sklearn, atol = 1e-5)

# Ridge Test
def test_Ridge():
	l = 0.1
	theta_Ridge = Ridge(X_train_s, y_train, l)
	predictions_Ridge = X_test_s @ theta_Ridge + y_offset

	predictions_Ridge_sklearn = Ridge_sklearn(alpha = l, fit_intercept=False).fit(X_train_s, y_train).predict(X_test_s) + y_offset
	
	assert np.allclose(predictions_Ridge.flatten(), predictions_Ridge_sklearn, atol = 1e-5)