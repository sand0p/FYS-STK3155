import autograd.numpy as np
import sys
from time import perf_counter

def Runge(x:np.ndarray)->np.ndarray:
    return 1 / (1 + 25*x**2)


#First the cost functions and their derivatives

def MSE(inputs, targets, layers, lam):
    return ((inputs-targets)**2).mean()

def MSE_der(inputs, targets):
    return 2 * (inputs - targets) / inputs.size

def MSE_L1(inputs, targets, layers, lam):
    l1_term = 0
    for W, b in layers:
        l1_term += np.sum(np.abs(W))
    return ((inputs-targets)**2).mean() + lam * l1_term

def MSE_L1_der(inputs, targets):
    return 2 * (inputs - targets) / inputs.size

def MSE_L2(inputs, targets, layers, lam):
    l2_term = 0
    for W, b in layers:
        l2_term += np.sum(W**2)
    return ((inputs-targets)**2).mean() + lam * l2_term

def MSE_L2_der(inputs, targets):
    return 2 * (inputs - targets) / inputs.size

def bce(inputs, targets, layers, lam):
    epsilon = 1e-12
    inputs = np.clip(inputs, epsilon, 1. - epsilon)
    return -np.mean(targets * np.log(inputs) + (1 - targets) * np.log(1 - inputs))

def bce_der(inputs, targets):
    epsilon = 1e-12
    inputs = np.clip(inputs, epsilon, 1. - epsilon)
    return -(targets / inputs - (1 - targets) / (1 - inputs)) / inputs.size

def softmax(z):
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def softmax_der(z):
    s = softmax(z)
    return s * (1 - s)

def cross_entropy_logits(inputs, targets, layers, lam):
    epsilon = 1e-12
    inputs = softmax(inputs)
    inputs = np.clip(inputs, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(inputs))

def cross_entropy_logits_der(inputs, targets):
    soft = softmax(inputs)
    return soft - targets


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels.astype(int)]


def cross_entropy(inputs, targets, layers, lam):
    epsilon = 1e-12  # Small constant to avoid log(0)
    inputs = np.clip(inputs, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(inputs))

def cross_entropy_der(inputs, targets):
    epsilon = 1e-12  # Small constant to avoid division by zero
    inputs = np.clip(inputs, epsilon, 1. - epsilon)
    return -targets / inputs

def accuracy_score(predictions, targets):
    return np.sum(predictions == targets) / len(targets)

def linear(z):
    return z

def linear_der(z):
    return 1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return np.exp(-z)/(1+np.exp(-z))**2

def ReLU(z):
    return np.where(z > 0, z, 0)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLU(z):
    negative_slope = 0.01
    return np.where(z > 0, z, negative_slope * z)

def leaky_ReLU_der(z):
    negative_slope = 0.01
    return np.where(z > 0, 1, negative_slope)

def OLS(X:np.ndarray, y:np.ndarray)->np.ndarray:
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def Ridge_Gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray, hyperparameter):
    n = X.shape[0]
    return 2 * ((1 / n) * X.T @ (X @ theta - y) + hyperparameter * theta)

def LASSO_Gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray, hyperparameter):
    n = X.shape[0]
    return (-2 / n) * X.T @ (y - X @ theta) + hyperparameter * np.sign(theta)

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
        lam=0.0,
        regularizer=None
    ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        # Ensure lengths match the number of layers
        if len(self.activation_funcs) != len(layer_output_sizes):
            raise ValueError("Number of activation functions must match the number of layers.")
        if len(self.activation_ders) != len(layer_output_sizes):
            raise ValueError("Number of activation derivatives must match the number of layers.")
        
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.lam = lam
        self.regularizer = regularizer
        self.layers = self._create_layers(network_input_size, layer_output_sizes)

    def _create_layers(self, network_input_size, layer_output_sizes):
        layers = []
        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = np.random.randn(layer_output_size, i_size)
            b = np.random.randn(layer_output_size)
            layers.append([W, b])
            i_size = layer_output_size
        return layers
    
    def predict_labels(self, inputs):
        outputs = []
        for input_sample in inputs:
            output = self.predict(input_sample)
            outputs.append(output)
        outputs = np.array(outputs)
        return np.argmax(outputs, axis=1)
    
    def predict(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = W @ a + b
            a = activation_func(z)
        return a
    
    def predict_batch(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W.T + b  # Adjusted for batch processing
            a = activation_func(z)
        return a

    #Want to calculate cost with one hot encoding for targets

    def cost(self, inputs, targets):
        return self.cost_fun(inputs, targets, self.layers, self.lam)
    
    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = W @ a + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def reset(self):
        self.layers = self._create_layers(self.network_input_size, self.layer_output_sizes)

    def compute_gradient(self, inputs, targets):
        '''
        Back propigation for finding gradients
        '''
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [[] for layer in self.layers]
        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]
            W, b = self.layers[i]
            if i == len(self.layers) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_der(predict, targets)
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                W_next, b_next = self.layers[i + 1]
                dC_da = dC_da @ W_next
            dC_dz = dC_da * activation_der(z)
            dC_dW = np.outer(dC_dz, layer_input)
            # Add regularization terms
            if self.regularizer == "L1":
                dC_dW += self.lam * np.sign(W)
            if self.regularizer == "L2":
                dC_dW += 2 * self.lam * W
            dC_db = dC_dz
            layer_grads[i] = [dC_dW, dC_db]
        return layer_grads

    def gradient_descent(self, training_inputs, training_targets, learning_rate, epochs = 2000):
        """
        Trains the network using gradient descent with fixed learning rate
        """
        n = len(training_inputs)
        bar = ProgressBar(epochs)
        for epoch in range(epochs):
            W_grads = [np.zeros(layer[0].shape) for layer in self.layers]
            b_grads = [np.zeros(layer[1].shape) for layer in self.layers]
            for training_input, training_target in zip(training_inputs, training_targets):
                grad = self.compute_gradient(training_input, training_target)
                for i, (dC_dW, dC_db) in enumerate(grad):
                    W_grads[i] += dC_dW
                    b_grads[i] += dC_db
            for i, (W_grad, b_grad) in enumerate(zip(W_grads, b_grads)):
                self.layers[i][0] -= learning_rate / n * W_grad
                self.layers[i][1] -= learning_rate / n * b_grad
            bar.step()
        bar.finish()

    def gradient_descent_stochastic(self, training_inputs, training_targets, learning_rate, epochs: int, minibatch_size: int):
        """
        Train the network with stochastic gradient descent
        Parameters:
            epochs (int): The number of passes through the entire dataset.
            minibatch_size (int): The size of each minibatch.
        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        np.random.seed(2025)
        n = len(training_inputs)
        bar = ProgressBar(epochs)
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            indexes = np.random.permutation(n)
            for j in range(0, n, minibatch_size):
                # Create minibatch
                batch_index = indexes[j:j+minibatch_size]
                W_grads = [np.zeros(layer[0].shape) for layer in self.layers]
                b_grads = [np.zeros(layer[1].shape) for layer in self.layers]
                training_batch = training_inputs[batch_index]
                target_batch = training_targets[batch_index]
                for training_input, training_target in zip(training_batch, target_batch):
                    grad = self.compute_gradient(training_input, training_target)
                    for i, (dC_dW, dC_db) in enumerate(grad):
                        W_grads[i] += dC_dW
                        b_grads[i] += dC_db
                for i, (W_grad, b_grad) in enumerate(zip(W_grads, b_grads)):
                    self.layers[i][0] -= learning_rate / len(training_batch) * W_grad
                    self.layers[i][1] -= learning_rate / len(training_batch) * b_grad
            bar.step()
        bar.finish()

    def RMSProp(self, training_inputs, training_targets, learning_rate, rho: float, epochs = 2000):
        """
        Performs RMSProp (Root Mean Square Propagation) gradient descent.
        Parameters:
            rho (float): The decay rate, a hyperparameter.
        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        delta = 1e-6
        n = len(training_inputs)
        bar = ProgressBar(epochs)
        for epoch in range(epochs):
            r_W = [np.zeros(layer[0].shape) for layer in self.layers]
            r_b = [np.zeros(layer[1].shape) for layer in self.layers]
            W_grads = [np.zeros(layer[0].shape) for layer in self.layers]
            b_grads = [np.zeros(layer[1].shape) for layer in self.layers]
            for training_input, training_target in zip(training_inputs, training_targets):
                grad = self.compute_gradient(training_input, training_target)
                for i, (dC_dW, dC_db) in enumerate(grad):
                    W_grads[i] += dC_dW
                    b_grads[i] += dC_db
                    r_W[i] = rho * r_W[i] + (1 - rho) * np.square(dC_dW)
                    r_b[i] = rho * r_b[i] + (1 - rho) * np.square(dC_db)
            for i, (W_grad, b_grad) in enumerate(zip(W_grads, b_grads)):
                self.layers[i][0] -= learning_rate / (n * np.sqrt(delta + r_W[i])) * W_grad
                self.layers[i][1] -= learning_rate / (n * np.sqrt(delta + r_b[i])) * b_grad
            bar.step()
        bar.finish()

    def RMSProp_stochastic(self, training_inputs, training_targets, learning_rate, rho: float, epochs, minibatch_size):
        """
        Performs RMSProp (Root Mean Square Propagation) gradient descent.
        Parameters:
            rho (float): The decay rate, a hyperparameter.
        Returns:
            np.ndarray: The optimized model parameters (theta).
        """
        np.random.seed(2025)
        delta = 1e-6
        n = len(training_inputs)
        bar = ProgressBar(epochs)
        for epoch in range(epochs):
            # print(f"Epoch {epoch}: Cost = {self.cost(self.predict_batch(training_inputs), training_targets)}")
            r_W = [np.zeros(layer[0].shape) for layer in self.layers]
            r_b = [np.zeros(layer[1].shape) for layer in self.layers]
            indexes = np.random.permutation(n)
            for j in range(0, n, minibatch_size):
                # Create minibatch
                batch_index = indexes[j:j+minibatch_size]
                batch_size = len(batch_index)
                W_grads = [np.zeros(layer[0].shape) for layer in self.layers]
                b_grads = [np.zeros(layer[1].shape) for layer in self.layers]
                training_batch = training_inputs[batch_index]
                target_batch = training_targets[batch_index]
                for training_input, training_target in zip(training_batch, target_batch):
                    grad = self.compute_gradient(training_input, training_target)
                    for i, (dC_dW, dC_db) in enumerate(grad):
                        W_grads[i] += dC_dW
                        b_grads[i] += dC_db
                        r_W[i] = rho * r_W[i] + (1 - rho) * dC_dW**2
                        r_b[i] = rho * r_b[i] + (1 - rho) * dC_db**2
                for i, (W_grad, b_grad) in enumerate(zip(W_grads, b_grads)):
                    self.layers[i][0] -= learning_rate / (batch_size * np.sqrt(delta + r_W[i])) * W_grad
                    self.layers[i][1] -= learning_rate / (batch_size * np.sqrt(delta + r_b[i])) * b_grad
            bar.step()
        bar.finish()

    def ADAM(self, training_inputs, training_targets, learning_rate, rho1, rho2, epochs = 2000):
        """
        """
        delta = 1e-6
        t = 0
        n = len(training_inputs)
        bar = ProgressBar(epochs)
        for epoch in range(epochs):
            s_W = [np.zeros(layer[0].shape) for layer in self.layers]
            r_W = [np.zeros(layer[0].shape) for layer in self.layers]
            s_b = [np.zeros(layer[1].shape) for layer in self.layers]
            r_b = [np.zeros(layer[1].shape) for layer in self.layers]
            W_grads = [np.zeros(layer[0].shape) for layer in self.layers]
            b_grads = [np.zeros(layer[1].shape) for layer in self.layers]
            for training_input, training_target in zip(training_inputs, training_targets):
                grad = self.compute_gradient(training_input, training_target)
                t += 1
                for i, (dC_dW, dC_db) in enumerate(grad):
                    s_W[i] = rho1 * s_W[i] + (1 - rho1) * dC_dW
                    s_b[i] = rho1 * s_b[i] + (1 - rho1) * dC_db
                    r_W[i] = rho2 * r_W[i] + (1 - rho2) * dC_dW**2
                    r_b[i] = rho2 * r_b[i] + (1 - rho2) * dC_db**2
                    s_W_hat = s_W[i] / (1 - rho1**t)
                    s_b_hat = s_b[i] / (1 - rho1**t)
                    r_W_hat = r_W[i] / (1 - rho2**t)
                    r_b_hat = r_b[i] / (1 - rho2**t)
                    W_grads[i] += s_W_hat / (np.sqrt(r_W_hat) + delta)
                    b_grads[i] += s_b_hat / (np.sqrt(r_b_hat) + delta)
            for i, (W_grad, b_grad) in enumerate(zip(W_grads, b_grads)):
                self.layers[i][0] -= learning_rate / n * W_grad
                self.layers[i][1] -= learning_rate / n * b_grad

            bar.step()  
        bar.finish()

    def ADAM_stochastic(self, training_inputs, training_targets, learning_rate, rho1, rho2, epochs, minibatch_size):
        # FIXED: Moved moments and t outside epoch loop
        delta = 1e-6
        n = len(training_inputs)
        t = 0
        np.random.seed(2025)
        bar = ProgressBar(epochs)
        for epoch in range(epochs):
            # print(f"Epoch {epoch}: Cost = {self.cost(self.predict_batch(training_inputs), training_targets)}")
            s_W = [np.zeros(layer[0].shape) for layer in self.layers]
            r_W = [np.zeros(layer[0].shape) for layer in self.layers]
            s_b = [np.zeros(layer[1].shape) for layer in self.layers]
            r_b = [np.zeros(layer[1].shape) for layer in self.layers]
            indexes = np.random.permutation(n)
            for j in range(0, n, minibatch_size):
                batch_index = indexes[j:j + minibatch_size]
                training_batch = training_inputs[batch_index]
                target_batch = training_targets[batch_index]
                batch_size = len(training_batch)
                W_grads = [np.zeros(layer[0].shape) for layer in self.layers]
                b_grads = [np.zeros(layer[1].shape) for layer in self.layers]
                for training_input, training_target in zip(training_batch, target_batch):
                    grad = self.compute_gradient(training_input, training_target)
                    t += 1
                    for i, (dC_dW, dC_db) in enumerate(grad):
                        s_W[i] = rho1 * s_W[i] + (1 - rho1) * dC_dW
                        s_b[i] = rho1 * s_b[i] + (1 - rho1) * dC_db
                        r_W[i] = rho2 * r_W[i] + (1 - rho2) * dC_dW**2
                        r_b[i] = rho2 * r_b[i] + (1 - rho2) * dC_db**2
                        s_W_hat = s_W[i] / (1 - rho1**t)
                        s_b_hat = s_b[i] / (1 - rho1**t)
                        r_W_hat = r_W[i] / (1 - rho2**t)
                        r_b_hat = r_b[i] / (1 - rho2**t)
                        W_grads[i] += s_W_hat / (np.sqrt(r_W_hat) + delta)
                        b_grads[i] += s_b_hat / (np.sqrt(r_b_hat) + delta)
                for i, (W_grad, b_grad) in enumerate(zip(W_grads, b_grads)):
                    self.layers[i][0] -= learning_rate / batch_size * W_grad
                    self.layers[i][1] -= learning_rate / batch_size * b_grad
            bar.step()
        bar.finish()


class ProgressBar:
    def __init__(self, width):
        self.width = min(width, 50)
        self.scale = self.width / width
        self.pos = 0
        self.steps = 0
        self.total_steps = width
        self.time = perf_counter()
        sys.stdout.write("[" + "-" * self.width + "]                                   ")
        sys.stdout.flush()
        sys.stdout.write("\r")
        sys.stdout.flush()
    def step(self):
        self.steps+=1
        if int(self.steps * self.scale)>self.pos:
            self.pos+=1
            time_rem = (perf_counter() - self.time) * (self.total_steps - self.steps) / self.steps
            bar = "[" + "#" * self.pos + "-" * (self.width - self.pos) + "] " + f"{time_rem:.2f}s remaining    "
            sys.stdout.write("\r" + bar)
            sys.stdout.flush()
    def finish(self):
        bar = "[" + "#" * self.width + "]\n"
        sys.stdout.write("\r" + bar)
        sys.stdout.flush()
        

class gradient_descent:
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