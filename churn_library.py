'''Helper functions to perform Churn prediction

If run as main script, the first argument points to a model parameter file
'''
import argparse
import json
import logging
from datetime import datetime as dt
from pathlib import Path
from sys import stdout

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

# Initializing Seaborn
sns.set()

# This logs to the logger of the script that import this module
# and invokes this function
logger = logging.getLogger(__name__)


class ChurnPredictor():
    """Class to perform Churn prediction
    """

    def __init__(self, conf: dict, model_params: dict):
        """Create a ChurnPredictor instance

        Parameters
        ----------
        conf : dict
            contains all configuration, like paths and data schema.
            Example specified in configs/config.json
        model_params : dict
            contains parameters for model training
            Example specified in configs/model_params.json
        """
        self.conf = conf
        self.model_params = model_params

        self.set_timestamp()

        self.data = None
        self.lrc = None
        self.cv_rfc = None
        self.x_full = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_full = None
        self.y_train_preds_lrc = None
        self.y_train_preds_rfc = None
        self.y_test_preds_lrc = None
        self.y_test_preds_rfc = None

    def set_timestamp(self):
        """Sets a new timestamp in the format "YYYY_mm_dd_HH_MM_ss"
        This string will be used to identify output files across a run
        """
        self.timestamp = dt.now().strftime("%Y_%m_%d_%H_%M_%s")

    @staticmethod
    def _validate_directory_path(path):
        """Validates that path refers to directory and creates if necessary

        Parameters
        ----------
        path : path-like
            str or Path object

        Returns
        -------
        Path


        Raises
        ------
        ValueError
            If given path is not directory, because it contains "." and
            refers to a file
        """
        path = Path(path)
        if not path.is_dir():
            if "." not in path.name:
                path.mkdir()
                logger.info("Created directory '%s'", path.absolute())
            else:
                exec_here = ValueError(f"Path {path} is not directory")
                logger.exception("", exc_info=exec_here)
                raise exec_here

        return path

    def import_data(self, raw_path):
        """Importing csv as pandas DataFrame

        Parameters
        ----------
        raw_path : str | Path
            Path to the csv file

        Returns
        -------
        pd.DataFrame
            Contains data plus a Churn column
        """
        path = Path(raw_path)
        raw_df = pd.read_csv(str(path.absolute()), index_col=0)

        # Only necessary to make pylint shut up
        self.data = pd.DataFrame(raw_df)

        logger.debug("Loaded DataFrame path %s", path)

        shape = self.data.shape
        logger.debug("DataFrame shape %s", shape)

        self.data.loc[:, "Churn"] = (
            self.data.loc[:, "Attrition_Flag"]
            .apply(
                lambda val: 0 if val == "Existing Customer" else 1
            )
        )

        logger.info("DataFrame established")

        return self.data

    def perform_eda(self, save_path):
        """Performs Exploratory Data Analysis and saves result images
        to save_path

        Parameters
        ----------
        save_path : path-like
            Path to directory where results will be saves

        Raises
        ------
        ValueError
            Raised when save_path is not a directory
        """
        save_path = self._validate_directory_path(save_path)

        logger.info("Saving images to %s", str(save_path))
        hist_cols = {
            "Churn": "count",
            "Customer_Age": "count",
            "Total_Trans_Ct": "count",
            "Marital_Status": "probability"
        }

        for col, stat in hist_cols.items():
            self._save_histplot(col, stat, save_path)
            logger.info("Created histplot for columns %s", col)

        self._save_heatmap(self.data.corr(), save_path)
        logger.info("Created correlation heatmap")

    def _save_histplot(self, col, stat, save_path):
        """Create and save a histogram plot

        Parameters
        ----------
        col : str
            Column name for histogram in self.data
        stat : "count", "probability" 
            Denotes the type of histogram
        save_path : path-like
            Path to directory where results will be saves
        """
        plt.figure(figsize=(20, 10))
        fig = sns.histplot(self.data[col], stat=stat).get_figure()
        fig.savefig(Path(save_path) / f"{self.timestamp}_histplot_{col}.png")

    def _save_heatmap(self, dfcorr, save_path):
        """Creates and saves a correlation heatmap

        Parameters
        ----------
        dfcorr : rectangular dataset
            Correlation DataFrame / 2D array, for example self.data.corr()
        save_path : path-like
            Path to directory where results will be saves
        """
        plt.figure(figsize=(20, 10))
        fig = sns.heatmap(
            dfcorr,
            annot=False,
            cmap='seismic',
            center=0.0,
            vmax=0.7,
            linewidths=2
        ).get_figure()

        fig.savefig(Path(save_path) / f"{self.timestamp}_corr_heatmap.png")

    def encoder_helper(self, category_list, response="Churn"):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook
        Modifies input df in-place, return is only for convenience!
        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for
        '''
        for col in category_list:
            self._encode_column(col, response)
            logger.info("Encoded column %s with average %s", col, response)

        return self.data

    def _encode_column(self, col, response):
        new_col_name = f"{col}_{response}"

        group_mean = (
            self.data
            .groupby(col)
            .mean()
            [response]
            .rename(new_col_name)
        )

        self.data.loc[:, new_col_name] = (
            self.data[col]
            .to_frame()
            .join(group_mean, on=col)
            [new_col_name]
        )

    def perform_feature_engineering(
        self,
        response,
        keep_cols,
        test_size=0.3,
        random_state=42,
        **kwargs,
    ):
        '''
        input:
                df: pandas dataframe
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                x_train: X training data
                x_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''
        self.x_full = self.data[keep_cols]
        self.y_full = self.data[response]
        self.x_train, self.x_test, self.y_train, self.y_test = (
            train_test_split(
                self.x_full,
                self.y_full,
                test_size=test_size,
                random_state=random_state,
                **kwargs,
            )
        )
        logger.info("Created train test split")
        return self.x_train, self.x_test, self.y_train, self.y_test

    def train_models(
        self,
        save_path=Path("./models"),
    ):
        '''
        train, store model results: images + scores, and store models
        input:
                x_train: X training data
                x_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        save_path = self._validate_directory_path(save_path)

        param_grid = self.model_params["param_grid"]
        random_state = self.model_params["random_state"]
        cv = self.model_params["cv"]

        rfc = RandomForestClassifier(random_state=random_state)

        self.cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=cv)
        self.cv_rfc.fit(self.x_train, self.y_train)
        logger.info("Performed Grid Search with parameters")
        for k, v in enumerate(param_grid):
            logger.info("\t%s:\t%s", k, v)

        self.lrc = LogisticRegression()
        self.lrc.fit(self.x_train, self.y_train)
        logger.info("Performed Logistic Regression")

        self.y_train_preds_rfc = self.cv_rfc.best_estimator_.predict(
            self.x_train
        )
        self.y_test_preds_rfc = self.cv_rfc.best_estimator_.predict(
            self.x_test
        )

        self.y_train_preds_lrc = self.lrc.predict(self.x_train)
        self.y_test_preds_lrc = self.lrc.predict(self.x_test)

        rfc_path = save_path / f'{self.timestamp}_rfc_model.pkl'
        lrc_path = save_path / f'{self.timestamp}_logistic_model.pkl'

        joblib.dump(self.cv_rfc.best_estimator_, rfc_path)
        logger.info("Saved Random Forest model to %s", rfc_path)

        joblib.dump(self.lrc, lrc_path)
        logger.info("Saved Logistic Regression model to %s", lrc_path)

        return self.cv_rfc, self.lrc

    def create_classification_reports(self, save_path):
        '''Creates images of Logistic Regression and Random Forest
        classification results

        Parameters
        ----------
        save_path : path-like
            Path where results will be saved to
        '''
        save_path = self._validate_directory_path(save_path)
        self._classification_report_image(
            self.y_train,
            self.y_train_preds_lrc,
            self.y_test,
            self.y_test_preds_lrc,
            "Logistic Regression",
            "lrc",
            save_path,
        )

        self._classification_report_image(
            self.y_train,
            self.y_train_preds_rfc,
            self.y_test,
            self.y_test_preds_rfc,
            "Random Forest",
            "rfc",
            save_path,
        )

    def _classification_report_image(
        self,
        y_train,
        y_train_preds,
        y_test,
        y_test_preds,
        model_name,
        model_abbrv,
        save_path,
    ):
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
        save_path = self._validate_directory_path(save_path)
        font_dict = {
            "fontsize": 10,
            "fontproperties": "monospace"
        }

        train_report = classification_report(y_train, y_train_preds)
        test_report = classification_report(y_test, y_test_preds)

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, f'{model_name} Train', font_dict)
        plt.text(0.01, 0.05, train_report, font_dict)
        plt.text(0.01, 0.6, f'{model_name} Test', font_dict)
        plt.text(0.01, 0.7, test_report, font_dict)
        plt.axis('off')
        img_path = save_path / \
            f"{self.timestamp}_{model_abbrv}_model_classification_report.png"
        plt.savefig(img_path)
        logger.info("Saved %s results to %s", model_name, img_path)

    def create_feature_importance_plots(self, save_path):
        """Creates images of features importances for
        Logistic Regression and Random Forest model

        Parameters
        ----------
        save_path : path-like
            Path where results will be saved to
        """
        save_path = self._validate_directory_path(save_path)
        
        self._feature_importance_plot(
            self.cv_rfc.best_estimator_,
            "Random Forest",
            "rfc",
            save_path
        )

    def _feature_importance_plot(
        self,
        model,
        model_name,
        model_abbrv,
        save_path,
    ):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                save_path: path to store the figure

        output:
                None
        '''
        save_path = self._validate_directory_path(save_path)

        # Calculate feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            logger.warning("""%s Model does not have feature importances.
                Skipping feature importance plot""", model_name)
            return

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self.x_test.columns[i] for i in indices]

        if len(names) != len(importances):
            raise ValueError("""Number of importances and found names
                dont match up""")

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title(f"{model_name} Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        xvals = range(len(importances))
        plt.bar(xvals, importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(xvals, names, rotation=90)
        plt.savefig(save_path /
                    f"{self.timestamp}_{model_abbrv}_feature_importance.png")

    def run_pipeline(self):
        """Runs the whole pipeline
        Expects self.conf, and self.model_params to be set and complete
        """
        self.import_data(
            self.conf["bank_data_path"]
        )

        self.perform_eda(self.conf["images_eda_path"])

        self.encoder_helper(
            self.conf["columns"]["cat"],
            response=self.conf["columns"]["response"]
        )

        self.perform_feature_engineering(
            self.conf["columns"]["response"],
            self.conf["columns"]["feature"],
        )

        self.train_models(
            save_path=self.conf["model_save_path"]
        )

        self.create_classification_reports(self.conf["images_results_path"])

        self.create_feature_importance_plots(self.conf["images_results_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config json file")
    parser.add_argument(
        "model_params",
        help="Path to the model parameter json file"
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fileobj:
        conf_main = json.load(fileobj)
    with open(args.model_params, encoding="utf-8") as fileobj:
        model_params_main = json.load(fileobj)

    streamhdlr = logging.StreamHandler(
        stream=stdout,
    )

    filehdlr = logging.FileHandler(
        filename=conf_main["log_path"],
        mode="a+",
    )
    logging.basicConfig(
        handlers=[streamhdlr, filehdlr],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level="INFO",
    )

    cp = ChurnPredictor(conf_main, model_params_main)

    cp.run_pipeline()
