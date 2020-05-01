import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def measure_performance(y_true, y_pred, average="macro"):
    measures = defaultdict(list)
    measures["Accuracy"].append(round(accuracy_score(y_true, y_pred),4))
    measures["Precision"].append(round(precision_score(y_true, y_pred, average=average),4))
    measures["Recall"].append(round(recall_score(y_true, y_pred, average=average),4))
    measures["F1"].append(round(f1_score(y_true, y_pred, average=average),4))
    measures["MCC"].append(round(matthews_corrcoef(y_true, y_pred),4))
    return pd.DataFrame.from_dict(measures)