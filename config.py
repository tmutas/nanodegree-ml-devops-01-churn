'''
Contains configuration constants,
like file paths or expected data column names and their types
'''
from pathlib import Path

BANK_DATA_PATH = Path("./data/bank_data.csv")

LOG_PATH = Path("./logs/log.txt")

TEST_LOG_PATH = Path("./logs/test_log.txt")

IMAGES_EDA_PATH = Path("./images/eda")

IMAGES_RESULTS_PATH = Path("./images/results")

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]
