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
        self.eps = self.y_series - self.yhat_series

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
    
    def plot_scatter(self, save_fig=False, root_name="generic_figure"):
        plt.close()
        plt.figure(figsize=(15,15))
        y_max0 = self.y_series.max()
        y_min0 = self.y_series.min()
        x_max0 = self.yhat_series.max()
        x_min0 = self.yhat_series.min()
        x_max_min0 = x_max0 - x_min0
        y_max_min0 = y_max0 - y_min0
        x_padding0 = 0.1 * x_max_min0
        y_padding0 = 0.1 * y_max_min0

        axis_min0 = min(x_min0 - x_padding0, y_min0 - y_padding0)
        axis_max0 = max(x_max0 + x_padding0, y_max0 + y_padding0)
        plt.axis('square')
        plt.rcParams["axes.grid"] = False

        plt.xlim([axis_min0, axis_max0])
        plt.ylim([axis_min0, axis_max0])
        plt.xlabel("Predicted value")
        plt.ylabel("Real value")
        plt.title(str(self.y_series.name)+"\n"+str(self.pretty_metrics()), fontsize=15)
        plt.scatter(self.yhat_series, self.y_series)
        plt.plot([axis_min0,axis_max0], [axis_min0,axis_max0], color='c')

        if save_fig==True:
            plt.savefig(str(root_name) + "0.svg", format="svg")

        plt.show()
    
    def plot_errors(self, save_fig=False, root_name="generic_figure"):
        plt.close()
        plt.figure(figsize=(15,15))
        x_max1 = self.yhat_series.max()
        x_min1 = self.yhat_series.min()
        y_max1 = self.eps.max()
        y_min1 = self.eps.min()
        x_max_min1 = x_max1 - x_min1
        y_max_min1 = y_max1 - y_min1
        x_padding1 = 0.1 * x_max_min1
        y_padding1 = 0.1 * y_max_min1

        axis_min1 = min(x_min1 - x_padding1, y_min1 - y_padding1)
        axis_max1 = max(x_max1 + x_padding1, y_max1 + y_padding1)
        
        plt.rcParams["axes.grid"] = False

        plt.scatter(self.yhat_series, self.eps)
        plt.plot([axis_min1,axis_max1], [0,0], 'c--')

        plt.xlim([x_min1 - x_padding1, x_max1 + x_padding1])
        plt.ylim([-axis_max1, axis_max1])
        plt.xlabel("Predicted value")
        plt.ylabel("Error")
        plt.title(self.y_series.name, fontsize=15)
        
        if save_fig==True:
            plt.savefig(str(root_name) + "1.svg", format="svg")

        plt.show()