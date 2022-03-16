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


def test_import():
    '''
    test data import
    '''
    try:
        df = ch.import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_data: The file wasn't found")
        #raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        for col in [*conf["columns"]["cat"], *conf["columns"]["quant"]]:
            assert col in df.columns
            logger.info("Testing import_data: Column %s is present", col)
    except AssertionError as err:
        logger.error("Testing import_data: Column %s not found!", col)
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df = ch.import_data("./data/bank_data.csv")

    with TemporaryDirectory() as tmpdir:
        ch.perform_eda(df, tmpdir)
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
                logger.info("Histplot for column %s exists", col)
            except AssertionError:
                logger.error("Histplot for column %s DOES NOT EXIST", col)
        try:
            assert any("_corr_heatmap.png" in fl for fl in os.listdir(tmpdir))
            logger.info("Correlation heatmap exists")
        except AssertionError:
            logger.error("Correlation Heatmap DOES NOT EXIST!")


def test_encoder_helper():
    '''
    test encoder helper
    '''
    pass


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    pass


def test_train_models():
    '''
    test train_models
    '''
    pass


if __name__ == "__main__":
    logger.warning("Start testing import_data")
    test_import()

    logger.warning("Start testing perform_eda")
    test_eda()

    logger.warning("All tests passed SUCCESSFULLY!")
