"""Testing and logging of functions in churn_library.py"""
import logging
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import churn_library as ch

with Path("configs/config.json").open(encoding="utf-8") as fil:
    conf = json.load(fil)
test_log_path = Path(conf["test_log_path"])

stream_handler = logging.StreamHandler(stream=sys.stdout)
file_handler = logging.FileHandler(test_log_path, mode="a+")
logging.basicConfig(
    handlers=[stream_handler, file_handler],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger()


def test_import(cp_test: ch.ChurnPredictor):
    """Test Importing data
    Parameters
    ----------
    cp_test : ch.ChurnPredictor
        ChurnPredictor object, initialized with parameters
    """
    try:
        df = cp_test.import_data(conf["bank_data_path"])
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError:
        logger.error("Testing import_data: The file wasn't found")
        #raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear"
            "to have rows and columns")
        raise err

    try:
        for col in [*conf["columns"]["cat"], *conf["columns"]["quant"]]:
            assert col in df.columns
            logger.info("Testing import_data: Column %s is present", col)
    except AssertionError as err:
        logger.error("Testing import_data: Column %s not found!", col)
        raise err


def test_eda(cp_test: ch.ChurnPredictor, tmpdir):
    """Test Perform Exploratory Data Analysis outputs

    Parameters
    ----------
    cp_test : ch.ChurnPredictor
        ChurnPredictor object, initialized with parameters
    tmpdir : path
        Path where the results are to be stored. Can be a temporary directory.
    """
    cp_test.import_data("./data/bank_data.csv")

    cp_test.perform_eda(tmpdir)
    hist_cols = {
        "Churn": "count",
        "Customer_Age": "count",
        "Total_Trans_Ct": "count",
        "Marital_Status": "probability"
    }
    # Testing that all histplots files exists in directory
    for col in hist_cols:
        target = f"_histplot_{col}.png"
        try:
            assert any(target in fl for fl in os.listdir(tmpdir))
            logger.info(
                "Testing perform_eda: Histplot for column %s exists", col)
        except AssertionError:
            logger.error(
                "Testing perform_eda: Histplot for column %s"
                "DOES NOT EXIST", col)
    try:
        assert any("_corr_heatmap.png" in fl for fl in os.listdir(tmpdir))
        logger.info("Testing perform_eda: Correlation heatmap exists")
    except AssertionError:
        logger.error(
            "Testing perform_eda: Correlation Heatmap DOES NOT EXIST!")


def test_encoder_helper(cp_test: ch.ChurnPredictor, cat_list, response):
    """Test Encode helper

    Parameters
    ----------
    cp_test : ch.ChurnPredictor
        ChurnPredictor object, initialized with parameters
    cat_list : list[str]
        List of category columns to be encoded in test
    response : str
        Name of the response column, used in encoding
    """
    cp_test.import_data(conf["bank_data_path"])

    logger.info("Testing encoder_helper")

    for col in cat_list:
        new_col_name = f"{col}_{response}"
        try:
            assert col in cp_test.data.columns
            logger.info(
                "Testing encoder_helper: Column %s to be encoded"
                "is present", col)

        except AssertionError:
            logger.error(
                "Testing encoder_helper: Column %s to be encoded"
                "is not present in data!", col)

        try:
            assert response in cp_test.data.columns
        except AssertionError:
            logger.error(
                "Testing encoder_helper: Response variable %s is"
                "not a column in data", col)

    logger.info("Running encoder_helper")

    cp_test.encoder_helper(cat_list)

    logger.info("Encoder helper has been run. Start asserting outcome")
    for col in cat_list:
        new_col_name = f"{col}_{response}"
        try:
            assert col in cp_test.data.columns
        except AssertionError:
            logger.error(
                "Testing encoder_helper: Column %s to be encoded"
                "is not present in data!", col)

        try:
            assert new_col_name in cp_test.data.columns
        except AssertionError:
            logger.error("Testing encoder_helper: Column %s was"
                         "not encoded!", col)


def test_perform_feature_engineering(
    cp_test: ch.ChurnPredictor,
    response,
    keep_cols,
    test_size
):
    """Test feature engineering and correct train test split

    Parameters
    ----------
    cp_test : ch.ChurnPredictor
        ChurnPredictor object, initialized with params and imported and encoded
        data
    response : str
        Name of the response column
    keep_cols : list[str]
        Feature columns to be kept for creating X data
    test_size : float
        Fraction of the full data to be split for test
    """

    logger.info("Testing perform_feature_engineering")

    cp_test.perform_feature_engineering(
        response,
        keep_cols,
        test_size
    )

    attr_to_test = [
        "x_full",
        "y_full",
        "x_train",
        "y_train",
        "x_test",
        "y_test"
    ]

    for attr in attr_to_test:
        try:
            assert hasattr(cp_test, attr)
            logger.info("Testing feature_eng: %s is present", attr)
        except AssertionError:
            logger.error("Testing feature_eng: %s was not created!", attr)

    x_test_len = len(cp_test.x_test)
    x_full_len = len(cp_test.x_full)
    x_train_len = len(cp_test.x_train)

    y_test_len = len(cp_test.y_test)
    y_full_len = len(cp_test.y_full)
    y_train_len = len(cp_test.y_train)

    try:
        assert x_test_len == y_test_len
        assert x_full_len == y_full_len
        assert x_train_len == y_train_len
        logger.info("Testing feature_eng: All x and y data sizes match")

    except AssertionError:
        logger.error("Testing feature_eng: x and y data sizes don't match!")

    try:
        assert x_test_len + x_train_len == x_full_len
        logger.info("Testing feature_eng: Train and Test data matches full!")

    except AssertionError:
        logger.error("Testing feature_eng: Train and Test data is not full!")

    min_test_len = (test_size - 0.01) * x_full_len
    max_test_len = (test_size + 0.01) * x_full_len

    try:
        assert min_test_len < x_test_len < max_test_len
        logger.info("Testing feature_eng: Test data matches give test_size!")
    except AssertionError:
        logger.error("Testing feature_eng: "
                     "Test data set is does not correspond to test_size")


def test_train_models(cp_test: ch.ChurnPredictor, tmpdir: Path):
    """Test Perform Exploratory Data Analysis outputs

    Parameters
    ----------
    cp_test : ch.ChurnPredictor
        ChurnPredictor object, initialized with parameters, and having been run
        the encoding and feature engineering
    tmpdir : path
        Path where the results are to be stored. Can be a temporary directory.
    """
    cp_test.train_models(tmpdir)

    try:
        assert any("rfc_model.pkl" in fl for fl in os.listdir(tmpdir))
        logger.info(
            "Testing train_models: RandomForest model is trained and pickled")
    except AssertionError:
        logger.error("Testing train_models: RandomForest model failed!")

    try:
        assert any("logistic_model.pkl" in fl for fl in os.listdir(tmpdir))
        logger.info(
            "Testing train_models: Logistic Regression model is trained"
            "and pickled")
    except AssertionError:
        logger.error("Testing train_models: Logistic Regression model failed!")


def test_create_model_reports(cp_test: ch.ChurnPredictor, tmpdir: Path):
    """Test creating model reports

    Parameters
    ----------
    cp_test : ch.ChurnPredictor
        ChurnPredictor object, initialized with parameters and a trained model
    tmpdir : path
        Path where the results are to be stored. Can be a temporary directory.
    """

    cp_test.create_model_reports(tmpdir)

    try:
        assert any(
            "rfc_model_classification_report.png"
            in fl for fl in os.listdir(tmpdir)
        )
        assert any(
            "lrc_model_classification_report.png"
            in fl for fl in os.listdir(tmpdir)
        )
        logger.info(
            "Testing model_reports: Classification reports created")
    except AssertionError:
        logger.error("Testing train_models: Classification reports failed!")

    try:
        assert any(
            "rfc_model_roc_curve.png"
            in fl for fl in os.listdir(tmpdir)
        )
        assert any(
            "lrc_model_roc_curve.png"
            in fl for fl in os.listdir(tmpdir)
        )
        logger.info(
            "Testing model_reports: ROC Curve plots created")
    except AssertionError:
        logger.error("Testing train_models: ROC Curve plots failed!")

    try:
        assert any(
            "feature_importance.png"
            in fl for fl in os.listdir(tmpdir)
        )
        logger.info(
            "Testing model_reports: Feature importance plot created")
    except AssertionError:
        logger.error("Testing train_models: Feature importance plot failed!")


if __name__ == "__main__":
    model_params = {
        "test_size": 0.3,
        "random_state": 42,
        "param_grid": {
            "n_estimators": [50, 100],
            "max_depth": [5, 50]
        },
        "cv": 5
    }

    cp_main = ch.ChurnPredictor(conf, model_params)
    logger.warning("Start testing import_data")
    test_import(cp_main)

    logger.warning("Start testing perform_eda")
    with TemporaryDirectory() as tmpdir_main:
        test_eda(cp_main, tmpdir_main)

    test_encoder_helper(
        cp_main,
        conf["columns"]["cat"],
        conf["columns"]["response"]
    )

    test_perform_feature_engineering(
        cp_main,
        conf["columns"]["response"],
        conf["columns"]["feature"],
        0.364
    )

    with TemporaryDirectory() as tmpdir_main:
        test_train_models(cp_main, tmpdir_main)

    with TemporaryDirectory() as tmpdir_main:
        test_create_model_reports(cp_main, tmpdir_main)

    logger.warning("All tests passed SUCCESSFULLY!")
