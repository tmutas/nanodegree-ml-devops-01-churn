'''Helper functions to perform Churn prediction'''
import logging
from pathlib import Path
import joblib
from datetime import datetime as dt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import config as conf

# Initializing Seaborn
sns.set()

# This logs to the logger of the script that import this module 
# and invokes this function
logger = logging.getLogger(__name__)

def import_data(raw_path):
    path = Path(raw_path)
    df = pd.read_csv(str(path.absolute()), index_col = 0)

    logger.debug("Loaded DataFrame path %s", path)

    shape = df.shape
    logger.debug("DataFrame shape %s", shape)

    df.loc[:,"Churn"] = (
        df.loc[:,"Attrition_Flag"]
        .apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
    )

    logger.info("DataFrame established")

    return df

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    hist_cols = {
        "Churn" : "count",
        "Customer_Age" : "count",
        "Total_Trans_Ct" : "count",
        "Marital_Status" : "probability"
    }

    for col, stat in hist_cols.items():
        _save_histplot(df, col, stat)
    
    _save_heatmap(df.corr())

def _save_histplot(df, col, stat):
    """_summary_

    Parameters
    ----------
    df : _type_
        _description_
    col : _type_
        _description_
    stat : _type_
        _description_
    """
    plt.figure(figsize=(20,10)) 
    fig = sns.histplot(df[col], stat=stat).get_figure()
    fig.savefig(conf.IMAGES_EDA_PATH / f"{dt.now()}_histplot_{col}.png")

def _save_heatmap(dfcorr):
    plt.figure(figsize=(20,10)) 
    fig = sns.heatmap(
        dfcorr,
        annot=False,
        cmap='seismic',
        center = 0.0,
        vmax=0.7,
        linewidths = 2
    ).get_figure()

    fig.savefig(conf.IMAGES_EDA_PATH / f"{dt.now()}_corr_heatmap.png")

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass