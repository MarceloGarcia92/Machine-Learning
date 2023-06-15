import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import class_likelihood_ratios, accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score

def reg_metrics(y_true, y_pred):
    """
    Calculate and return the root mean squared error.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def binary_class_metrics(y_true, y_pred):
    likelihood = class_likelihood_ratios(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return likelihood, acc

def multi_class_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    return f1, recall, precision
