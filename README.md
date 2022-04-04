# Predict Customer Churn - Course Project

## Project Description

- The Project **Predict Customer Churn** was part of the ML DevOps Engineer Nanodegree on Udacity
- The task was to refactore exploratory ML code from a notebook into a separate, maintainable Python module, adhering to PEP8 standards, as well as showing basics of logging and tests.


## Project Organization

The project is structured into one file `churn_library.py` containing the main code in a class. Configuration is expected in two `.json` files, one for the model hyperparameters, one for the preprocessing and I/O. The raw data is stored in the (surprise!) `data/` directory.

Dependencies are detailed in `requirements.txt`

    ├── README.md                             <- The top-level README you are reading right now, explaining the project
    ├── requirements.txt                      <- Detailing Python package dependencies
    ├── Guide.ipynb                           <- Explains the task for the given project
    ├── churn_notebook.ipynb                  <- The original notebook to be refactored for this project
    ├── churn_library.py                      <- The model training code from churn_notebook.ipynb refactored into a class. Can also be run as a main script. 
    ├── churn_script_logging_and_tests.py     <- The runs and tests all functions in churn_library.py and logs the results
    │
    ├── data
    │   └── bank_data.py            <- The original, provided raw data file
    │
    ├── configs                     <- Example configuration files, recommended to store custom ones there too.
    │   ├── config.json             <- Contains desired save paths, data paths and data schema
    │   └── model_params.json       <- Contains hyperparameters regarding the model training

## Running tests
In the root directory of the project, simply invoke the test script:
```python churn_script_logging_and_tests.py```

It will output the log and current progress into the console, as well as into the `test_log_path` directory provided in `configs/config.json`.
The output of running the main code are stored in temporary directory and will be deleted at the end of the tests, so running the test will not leave any traces, except for the log.

## Running Files
The `churn_library.py` can be invoked directly and will run the entire training pipeline. It expects two parameters, path to the config file and path to the model parameters.
Example files are provided in the repository in `configs/`

## Author
Timo Mutas


