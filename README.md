# Predict Customer Churn - Course Project

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity
- The task was to refactore exploratory ML code from a notebook into a separate, maintainable Python module, as well as showing basics of logging and tests.

## Project Description
The project is structure into one file "churn_library.py" containing the main code in a class. Configuration is expected in two .json files, one for the model hyperparameters, one for the preprocessing and I/O. The raw data is stored in the (surprise!) data/ directory.

## Running tests
In the root directory of the project, simply invoke the test script:
"python churn_script_logging_and_tests.py´
It will output the log and current progress into the console, as well as into the "test_log_path" directory provided in configs/config.json.
The output of running the main code are stored in temporary directory and will be deleted at the end of the tests, so running the test will not leave any traces, except for the log.

## Running Files
The churn_library.py can be invoked directly and will run the entire training pipeline. It expects two parameters, path to the config file and path to the model parameters.
Example files are provided in the repository in configs/

## Author
Timo Mutas


