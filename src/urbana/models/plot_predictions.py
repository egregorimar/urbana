import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

class PredictedAccuracy:
    def __init__(self, y_series, yhat_series):
        if isinstance(yhat_series, np.ndarray):
            yhat_series = pd.Series(yhat_series, name=f"predicted {y_series.name}")
            yhat_series.index = y_series.index

        self.y_series = y_series
        self.yhat_series = yhat_series

    @staticmethod
    def regression_accuracy_metrics(y, yhat):
        """Metrics to evaluate the accuracy of regression models.

        Ref: https://www.datatechnotes.com/2019/10/accuracy-check-in-python-mae-mse-rmse-r.html
        """
        mse = metrics.mean_squared_error(y, yhat)

        metrics_dict = {
            "MAE": metrics.mean_absolute_error(y, yhat),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "r2": metrics.r2_score(y, yhat),
        }

        return metrics_dict

    def metrics(self):
        return PredictedAccuracy.regression_accuracy_metrics(
            self.y_series, self.yhat_series
        )

    def pretty_metrics(self, decimals: int = 2, separation_string: str = ", "):
        return separation_string.join(
            [
                f"{k}: {round(v, decimals):.{decimals}f}"
                for k, v in self.metrics().items()
            ]
        )
    
    def plot_scatter(self, main_title="Actual vs predicted measure"):
        y_max = self.y_series.max()
        y_min = self.y_series.min()
        x_max = self.yhat_series.max()
        x_min = self.yhat_series.min()
        x_max_min = x_max - x_min
        y_max_min = y_max - y_min
        x_padding = 0.1 * x_max_min
        y_padding = 0.1 * y_max_min

        axis_min = min(x_min - x_padding, y_min - y_padding)
        axis_max = max(x_max + x_padding, y_max + y_padding)

        plt.scatter(self.yhat_series, self.y_series)
        plt.plot([axis_min,axis_max], [axis_min,axis_max], color='c')

        plt.xlim([axis_min, axis_max])
        plt.ylim([axis_min, axis_max])
        plt.xlabel(str(self.yhat_series.name))
        plt.ylabel(str(self.y_series.name))
        plt.title(str(main_title)+"\n"+str(self.pretty_metrics()), fontsize=10)
