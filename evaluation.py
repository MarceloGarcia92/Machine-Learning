import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import class_likelihood_ratios, accuracy_score

def reg_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def binary_class_metrics(y_true, y_pred):
    likelihood = class_likelihood_ratios(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return likelihood, acc
