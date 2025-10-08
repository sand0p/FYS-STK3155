import numpy as np

def Runge(x:np.ndarray)->np.ndarray:
    """
    Calculates the Runge function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: The value of the Runge function for each element in x.
    """
    return 1 / (1 + 25*x**2)

def OLS(X:np.ndarray, y:np.ndarray)->np.ndarray:
    """
    Performs Ordinary Least Squares (OLS) regression using the normal equation.

    Parameters:
        X (np.ndarray): Design matrix of shape (n_samples, n_features).
        y (np.ndarray): Target values of shape (n_samples,).

    Returns:
        np.ndarray: The optimal parameters (beta) of shape (n_features,).
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def Ridge(X:np.ndarray, y:np.ndarray, l)->np.ndarray:
    """
    Performs Ridge regression using the analytical solution.

    Parameters:
        X (np.ndarray): Design matrix of shape (n_samples, n_features).
        y (np.ndarray): Target values of shape (n_samples,).
        l (float): Hyperparameter lambda (regularization strength).

    Returns:
        np.ndarray: The optimal parameters (beta) of shape (n_features,).
    """
    return np.linalg.pinv(X.T @ X + l * np.identity(len(X[0]))) @ X.T @ y

def OLS_Gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray, **kwargs):
    """
    Calculates the gradient of the Ordinary Least Squares (OLS) cost function.

    Parameters:
        X (np.ndarray): Design matrix.
        y (np.ndarray): Target values.
        theta (np.ndarray): Current model parameters.

    Returns:
        np.ndarray: The gradient of the cost function with respect to theta.
    """
    n = X.shape[0]
    return (2 / n) * X.T @ (X @ theta - y)

def Ridge_Gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray, hyperparameter):
    """
    Calculates the gradient of the Ridge regression cost function.

    Parameters:
        X (np.ndarray): Design matrix.
        y (np.ndarray): Target values.
        theta (np.ndarray): Current model parameters.
        hyperparameter (float): The regularization parameter lambda.

    Returns:
        np.ndarray: The gradient of the cost function with respect to theta.
    """
    n = X.shape[0]
    return 2 * ((1 / n) * X.T @ (X @ theta - y) + hyperparameter * theta)

def LASSO_Gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray, hyperparameter):
    """
    Calculates the gradient of the LASSO regression cost function.

    Parameters:
        X (np.ndarray): Design matrix.
        y (np.ndarray): Target values.
        theta (np.ndarray): Current model parameters.
        hyperparameter (float): The regularization parameter lambda.

    Returns:
        np.ndarray: The gradient of the cost function with respect to theta.
    """
    n = X.shape[0]
    return (-2 / n) * X.T @ (y - X @ theta) + hyperparameter * np.sign(theta)

class gradient_descent:
    """
    A class that implements various gradient descent optimization algorithms.

    This class provides methods for standard, stochastic, and adaptive
    gradient descent techniques with Momentum, AdaGrad, RMSProp, and ADAM.

    Attributes:
        X (np.ndarray): The design matrix.
        y (np.ndarray): The target values.
        n (int): The number of data points.
        polynomial_degree (int): The number of features (degree of polynomial).
        learning_rate (float): The learning rate for the optimizer.
        hyperparam (float, optional): The regularization hyperparameter (lambda).
        gradient (callable): The function to compute the gradient of the cost function.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, gradient: callable, learning_rate, hyperparameter = None):
        """
        Initializes the gradient_descent optimizer.

        Parameters:
            X (np.ndarray): The design matrix.
            y (np.ndarray): The target values.
            gradient (callable): The function for gradient computation.
            learning_rate (float): The learning rate.
            hyperparameter (float, optional): Regularization parameter. Defaults to None.
        """
        self.X = X
        self.y = y.flatten()
        self.n = X.shape[0]
        self.polynomial_degree = X.shape[1]
        self.learning_rate = learning_rate
        self.hyperparam = hyperparameter
        self.gradient = gradient  

    def gradient_descent(self):
            """
            Performs standard gradient descent.

            Returns:
                np.ndarray: The optimized model parameters (theta).
            """
            theta = np.zeros(self.polynomial_degree)
            prev_theta = theta.copy() + 1
            num_iters = 0
            while stopping_parameter(num_iters, prev_theta, theta) == False:
                prev_theta = theta.copy()
                g = self.gradient(self.X, self.y, theta, hyperparameter = self.hyperparam)
                theta -= self.learning_rate * g
                num_iters += 1
            return theta 

    def gradient_descent_stochastic(self, epochs: int, minibatch_size: int):
        """
        Performs stochastic gradient descent (SGD) with minibatches.

        Parameters:
            epochs (int): The number of passes through the entire dataset.
            minibatch_size (int): The size of each minibatch.

        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        np.random.seed(2025)
        theta = np.zeros(self.polynomial_degree)
        
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            indexes = np.random.permutation(self.n)
            for i in range(0, self.n, minibatch_size):
                # Create minibatch
                batch_index = indexes[i:i+minibatch_size]
                
                X_batch = self.X[batch_index]
                y_batch = self.y[batch_index]
                g = self.gradient(X_batch, y_batch, theta, hyperparameter = self.hyperparam)
                theta -= self.learning_rate * g
                
        return theta
    
    def momentum(self, alpha):
        """
        Performs gradient descent with momentum.

        Parameters:
            alpha (float): The momentum parameter, typically close to 1.

        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        v = np.zeros(self.polynomial_degree)
        theta = np.zeros(self.polynomial_degree)
        prev_theta = theta.copy() + 1
        num_iters = 0
        
        while stopping_parameter(num_iters, prev_theta, theta) == False:
            prev_theta = theta.copy()
            g = self.gradient(self.X, self.y, theta, hyperparameter = self.hyperparam)
            v = alpha * v - self.learning_rate * g
            theta += v
            num_iters += 1
            
        return theta
    
    def momentum_stochastic(self, alpha, epochs: int, minibatch_size: int):
        """
        Performs stochastic gradient descent with momentum.

        Parameters:
            alpha (float): The momentum parameter, typically close to 1.
            epochs (int): The number of passes through the entire dataset.
            minibatch_size (int): The size of each minibatch.

        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        np.random.seed(2025)
        v = np.zeros(self.polynomial_degree)
        theta = np.zeros(self.polynomial_degree)
        
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            indexes = np.random.permutation(self.n)
            for i in range(0, self.n, minibatch_size):
                # Create minibatch
                batch_index = indexes[i:i+minibatch_size]
                
                X_batch = self.X[batch_index]
                y_batch = self.y[batch_index]
                g =  self.gradient(X_batch, y_batch, theta, hyperparameter = self.hyperparam)
                v = alpha * v - self.learning_rate * g
                theta += v
        return theta
    
    def AdaGrad(self):
        """
        Performs AdaGrad gradient descent.

        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        delta = 1e-6
        r = np.zeros(self.polynomial_degree)
        theta = np.zeros(self.polynomial_degree)
        prev_theta = theta.copy() + 1
        num_iters = 0
        
        while stopping_parameter(num_iters, prev_theta, theta) == False:
            prev_theta = theta.copy()
            g = self.gradient(self.X, self.y, theta, hyperparameter = self.hyperparam)
            r += g * g
            theta += -self.learning_rate / (delta + r ** 0.5) * g
            num_iters += 1
            
        return theta
    
    def AdaGrad_stochastic(self, epochs: int, minibatch_size: int):
        """
        Performs stochastic AdaGrad gradient descent.

        Parameters:
            epochs (int): The number of passes through the entire dataset.
            minibatch_size (int): The size of each minibatch.

        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        np.random.seed(2025)
        delta = 1e-6
        theta = np.zeros(self.polynomial_degree)
        
        for epoch in range(epochs):
            r = np.zeros(self.polynomial_degree)
            # Shuffle the data at the beginning of each epoch
            indexes = np.random.permutation(self.n)
            for i in range(0, self.n, minibatch_size):
                # Create minibatch
                batch_index = indexes[i:i+minibatch_size]
                
                X_batch = self.X[batch_index]
                y_batch = self.y[batch_index]
                g = self.gradient(X_batch, y_batch, theta, hyperparameter = self.hyperparam)
                r += g * g
                theta += -self.learning_rate / (delta + r ** 0.5) * g
        return theta
    
    def RMSProp(self, rho: float):
        """
        Performs RMSProp (Root Mean Square Propagation) gradient descent.

        Parameters:
            rho (float): The decay rate, a hyperparameter.

        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        delta = 1e-6
        r = np.zeros(self.polynomial_degree)
        theta = np.zeros(self.polynomial_degree)
        prev_theta = theta.copy() + 1
        num_iters = 0
        
        while stopping_parameter(num_iters, prev_theta, theta) == False:
            prev_theta = theta.copy()
            g = self.gradient(self.X, self.y, theta, hyperparameter = self.hyperparam)
            r = rho * r + (1 - rho) * g * g
            theta += -self.learning_rate / ( (delta + r)**0.5 ) * g
            num_iters+=1
            
        return theta
    
    def RMSProp_stochastic(self, rho: float, epochs: int, minibatch_size: int):
        """
        Performs stochastic RMSProp gradient descent.

        Parameters:
            rho (float): The decay rate.
            epochs (int): The number of passes through the entire dataset.
            minibatch_size (int): The size of each minibatch.

        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        np.random.seed(2025)
        delta = 1e-6
        theta = np.zeros(self.polynomial_degree)
        
        for epoch in range(epochs):
            # Reset r for each epoch
            r = np.zeros(self.polynomial_degree)

            # Shuffle the data at the beginning of each epoch
            indexes = np.random.permutation(self.n)

            for i in range(0, self.n, minibatch_size):
                # Create minibatch
                batch_index = indexes[i:i+minibatch_size]

                X_batch = self.X[batch_index]
                y_batch = self.y[batch_index]
                g = self.gradient(X_batch, y_batch, theta, hyperparameter = self.hyperparam)
                r = rho * r + (1 - rho) * g * g
                theta += -self.learning_rate / ( (delta + r)**0.5 ) * g

        return theta

    def ADAM(self, rho1, rho2):
        """
        Performs ADAM (Adaptive Moment Estimation) gradient descent.

        Parameters:
            rho1 (float): The decay rate for the first moment.
            rho2 (float): The decay rate for the second moment.

        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        delta = 1e-6
        s = np.zeros(self.polynomial_degree); r = np.zeros(self.polynomial_degree)
        t = 0
        theta = np.zeros(self.polynomial_degree)
        prev_theta = theta.copy() + 1
        num_iters = 0

        while stopping_parameter(num_iters, prev_theta, theta) == False:
            prev_theta = theta.copy()
            g = self.gradient(self.X, self.y, theta, hyperparameter = self.hyperparam)
            t += 1
            s = rho1 * s + (1 - rho1) * g
            r = rho2 * r + (1 - rho2) * g * g
            s_hat = s / (1 - rho1 ** t)
            r_hat = r / (1 - rho2 ** t)
            theta += -self.learning_rate * s_hat / (delta + r_hat ** 0.5)
            num_iters += 1

        return theta

    def ADAM_stochastic(self, rho1: float, rho2: float, epochs: int, minibatch_size: int):
        """
        Performs stochastic ADAM gradient descent.

        Parameters:
            rho1 (float): The decay rate for the first moment.
            rho2 (float): The decay rate for the second moment.
            epochs (int): The number of passes through the entire dataset.
            minibatch_size (int): The size of each minibatch.

        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        np.random.seed(2025)
        delta = 1e-6
        t = 0
        theta = np.zeros(self.polynomial_degree)
        
        for epoch in range(epochs):
            # Reset s and r for each epoch
            s = np.zeros(self.polynomial_degree); r = np.zeros(self.polynomial_degree)
            t += 1

            # Shuffle the data at the beginning of each epoch
            indexes = np.random.permutation(self.n)
            for i in range(0, self.n, minibatch_size):
                # Create minibatch
                batch_index = indexes[i:i+minibatch_size]

                X_batch = self.X[batch_index]
                y_batch = self.y[batch_index]
                g = self.gradient(X_batch, y_batch, theta, hyperparameter = self.hyperparam)
                s = rho1 * s + (1 - rho1) * g
                r = rho2 * r + (1 - rho2) * g * g
                s_hat = s / (1 - rho1 ** t)
                r_hat = r / (1 - rho2 ** t)
                theta += -self.learning_rate * s_hat / (delta + r_hat ** 0.5)

        return theta

def stopping_parameter(n: int, prev_theta: np.ndarray, theta: np.ndarray) -> bool:
    """
    Determines the stopping condition for gradient descent iterations.

    Stops after a maximum number of iterations (1000) or if the sum of absolute differences in parameters
    is below a tolerance threshold (1e-4).

    Parameters:
        n (int): The current number of iterations.
        prev_theta (np.ndarray): The parameter values from the previous iteration.
        theta (np.ndarray): The current parameter values.

    Returns:
        bool: True if the optimization should stop, False otherwise.
    """
    max_iters = 1000
    gradient_tolerance = 1e-4

    # Using sum of absolute differences for tolerance
    delta = np.abs(prev_theta - theta).sum()

    if n > max_iters or delta < gradient_tolerance:
        return True
    return False
