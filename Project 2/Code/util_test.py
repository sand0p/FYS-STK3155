from util import *
from autograd import elementwise_grad, jacobian
import numpy as np

def test_sigmoid():
	z = np.random.randn(1000) * 100
	own_sigmoid = sigmoid_der(z)

	ag_sigmoid_der = elementwise_grad(sigmoid, 0)
	ag_sigmoid = ag_sigmoid_der(z)

	assert np.allclose(own_sigmoid, ag_sigmoid, rtol = 1e-5), "Sigmoid derivatives not equal"

def test_ReLU():
	z = np.random.randn(1000) * 100
	own_ReLU = ReLU_der(z)

	ag_ReLU_der = elementwise_grad(ReLU, 0)
	ag_ReLU = ag_ReLU_der(z)

	assert np.allclose(own_ReLU, ag_ReLU, rtol = 1e-5), "ReLU derivatives not equal"

def test_leaky_ReLU():
	z = np.random.randn(1000) * 100
	own_leaky_ReLU = ReLU_der(z)

	ag_leaky_ReLU_der = elementwise_grad(ReLU, 0)
	ag_leaky_ReLU = ag_leaky_ReLU_der(z)

	assert np.allclose(own_leaky_ReLU, ag_leaky_ReLU, rtol = 1e-5), "Leaky ReLU derivatives not equal"

def test_MSE():
	y_true = np.random.randn(1000)
	y_pred = np.random.randn(1000)
	own_mse = MSE_der(y_true, y_pred)

	ag_mse = elementwise_grad(MSE, 0)
	ag_mse_val = ag_mse(y_true, y_pred, [], lam = 0.01)
	assert np.allclose(own_mse, ag_mse_val, rtol = 1e-5), "MSE derivatives not equal"

def test_MSE_L1():
	y_true = np.random.randn(1000)
	y_pred = np.random.randn(1000)
	own_mse_L1 = MSE_L1_der(y_true, y_pred)

	ag_mse_L1 = elementwise_grad(MSE_L1, 0)
	ag_mse_L1_val = ag_mse_L1(y_true, y_pred, [], lam = 0.01)

	assert np.allclose(own_mse_L1, ag_mse_L1_val, rtol = 1e-5), "MSE L1 derivatives not equal"

def test_MSE_L2():
	y_true = np.random.randn(1000)
	y_pred = np.random.randn(1000)
	own_mse_L2 = MSE_L2_der(y_true, y_pred)

	ag_mse_L2 = elementwise_grad(MSE_L2, 0)
	ag_mse_L2_val = ag_mse_L2(y_true, y_pred, [], lam = 0.01)

	assert np.allclose(own_mse_L2, ag_mse_L2_val, rtol = 1e-5), "MSE L2 derivatives not equal"

def test_BCE():
	targets = np.random.randint(0, 2, size = 1000)
	inputs = np.random.rand(1000)
	own_bce = bce_der(inputs, targets)

	ag_bce = elementwise_grad(bce, 0)
	ag_bce_val = ag_bce(inputs, targets, [], lam = 0.01)

	assert np.allclose(own_bce, ag_bce_val, rtol = 1e-5), "BCE derivatives not equal"

def test_softmax():
	z = np.random.randn(1000) * 100
	own_softmax = softmax_der(z)

	ag_softmax_der = jacobian(softmax)
	ag_softmax = np.diag(ag_softmax_der(z))
	assert np.allclose(own_softmax, ag_softmax, rtol = 1e-5), "Softmax derivatives not equal"

