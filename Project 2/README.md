# Feed-Forward Neural Network implementation and optimization
By Sander Oppen and Ole Markus Bråthen.

We implement and explore the use of neural networks for regression and classification tasks. Regression performance is measured and compared against OLS regression on the Runge function. The neural network is also used on the MNIST dataset for multiclass classification of handwritten digits, and compared against the implementations of PyTorch.  

The project uses python3.13. To install the required packages, run the command
```
pip install -r requirements.txt
```

All code can be found in the `Code` folder. `util.py` contains all of our own implementations of the neural networks and functions. Tests for the correctness of derivatives can be found in `util_test.py`. It is recomended to use pytest for running all test functions.

Code for experiments and results plotting is organized in jupyter notebooks. For regression with neural networks, `nn_regression` contains basic NN regression and some learning rate analysis. `architectures` contains experiments on different network architectures, and `regularization` contains experiments on the use of L1/L2 regularizers.
The `MNIST`-notebook contains code for the configuration search for classification. The optimal setups found in these experiments are used in the `optimal_MNIST` notebook for further experiments.

All figures used are provided in the `figs`-folder. Two json files containing the results of the configuration search are also provided.
