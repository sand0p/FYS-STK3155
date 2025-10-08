# Linear regression and resampling techniques on the Runge function
By Sander Oppen and Ole Markus Bråthen.

We explore how OLS, Ridge, and LASSO regression perform on the one-dimensional Runge function, as well as their dependence on hyperparameters. The regression fitting is done through both exact analytical solutions with matrix inversion, and with different gradient descent methods. In addition, we explore resampling techniques to understand the bias-variance tradeoff.

The project uses python3.13. To install the required packages, run the command
```
pip install -r requirements.txt
```

All code can be found in the `Code` folder. `util.py` contains all of our own implementations of the algorithms we have used. All the results for each subsection of *results and discussion* is given in a seperate jupyter notebook. 