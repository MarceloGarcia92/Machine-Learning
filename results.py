from numpy import sqrt
from sklearn.metrics import mean_squared_error


def predict_and_evaluate(model, x_test, y_test):
    """
    Predict values from test set, calculate and return the root mean squared error.
    """
    y_pred = model.predict(x_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))    
    return rmse