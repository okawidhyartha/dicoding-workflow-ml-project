import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss, roc_auc_score
import time
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import warnings

# load the dataset
df_train = pd.read_csv("./diabetes_preprocessing/diabetes_train.csv")
df_test = pd.read_csv("./diabetes_preprocessing/diabetes_test.csv")

# get X and y features
X_train = df_train.drop(columns=["CLASS"])
y_train = df_train["CLASS"]
X_test = df_test.drop(columns=["CLASS"])
y_test = df_test["CLASS"]

warnings.filterwarnings("ignore")

with mlflow.start_run():
    # create the model
    rf = RandomForestClassifier(random_state=42)
    
    # set start time
    start_time = time.time()
    
    # fit the model
    model = rf.fit(X_train, y_train)
    
    # set end time
    end_time = time.time()

    # make predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)

    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    logloss = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    score = model.score(X_test, y_test)

    # log parameters and metrics
    # mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", report["1"]["f1-score"])
    mlflow.log_metric("recall", report["1"]["recall"])
    mlflow.log_metric("precision", report["1"]["precision"])
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("score", score)
    mlflow.log_metric("training_time", end_time - start_time)
    
    input_example = X_test.iloc[:1]
    mlflow.sklearn.log_model(model, "model", input_example=input_example)
    
    # log confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    
    random_int = np.random.randint(0, 100)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/confusion_matrix_{random_int}.png")
    plt.close()  # Close the plot to avoid displaying it
    mlflow.log_artifact(f"plots/confusion_matrix_{random_int}.png")
