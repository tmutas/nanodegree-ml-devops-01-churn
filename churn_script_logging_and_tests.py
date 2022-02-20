import logging

import churn_library as ch
import config as conf

#TODO: Change to proper filename
logging.basicConfig(
    filename=str(conf.TEST_LOG_PATH.absolute()),
    level = logging.DEBUG,
    filemode='a+',
    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
logger.setLevel("DEBUG")

def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = ch.import_data("./data/bank_data copy.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        for col in [*conf.CAT_COLUMNS, *conf.QUANT_COLUMNS]:
            assert col in df.columns
            logger.info("Testing import_data: Column %s is present", col)
    except AssertionError as err:
        logger.error("Testing import_data: Column %s not found", col)
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df = ch.import_data("./data/bank_data.csv")

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








