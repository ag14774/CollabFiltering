# Collaborative Filtering for CIL

## Getting Started

### Prerequisites

All prerequisites are included in the file `requirements.yml`

To install using `conda`:
```
conda env create --file requirements.yml
conda activate CILTest
```

### How to run

First, let's get familiar with the CLI arguments:
```
$ ./main.py --help
usage: main.py [-h] --training-data TRAINING_DATA --submission-data
               SUBMISSION_DATA [--validation-data VALIDATION_DATA]
               --model-conf MODEL_CONF [--verbose] [--store-in-ensemble]
               [--no-output] [--matrix-rows MATRIX_ROWS]
               [--matrix-columns MATRIX_COLUMNS]

Collaborative Filtering software for CIL

optional arguments:
  -h, --help            show this help message and exit
  --training-data TRAINING_DATA
                        File to read training data from
  --submission-data SUBMISSION_DATA
                        File to read submission data from
  --validation-data VALIDATION_DATA
                        File to read validation data from
  --model-conf MODEL_CONF
                        File to read the model configuration
  --verbose, -v
  --store-in-ensemble   Set this to true if you want to store the total matrix
                        for later use in ensemble
  --no-output           Disable all output
  --matrix-rows MATRIX_ROWS
                        Rows for the matrix
  --matrix-columns MATRIX_COLUMNS
                        Columns for the matrix
```

To run our first experiment we need to create a `conf.json` file with all the information and settings of our model:
```
{
  "predictor": "svd_simple",
  "CV_folds": 5,
  "param_list": [
    {
      "knew": 8
    },
    {
      "knew": 10
    },
    {
      "knew": 12
    },
    {
      "knew": 14
    }
  ],
  "baseline_method": "mean_col_baseline",
  "baseline_settings": {
  },
  "rounding_threshold": 0.0
}
```
This file basically says that we want to use a simple SVD procedure to predict the ratings. `svd_simple` is the name of the function in `collabfilter.py`. Next, we need to include the number of folds to use for cross-validation(in this case 5 with 0 meaning no CV) and a list of parameters to test. In our case, we will attempt to make predictions by keeping the first 8, 10, 12 and 14 singular values. The best one will be automatically chosen and rerun using the whole training set. We also provide an imputation method. Here we just impute by taking the mean of each column. Some imputation methods take additional parameters and those can be set in `baseline_settings`. Finally, we have the option to round any number to the closest integer, if the distance to the integer is less than `rounding_threshold`. When set to 0.0 rounding is disabled.

Finally we can run it using:
```
./main.py --training-data data_train.csv --submission-data sampleSubmission.csv --model-conf conf.json
```

Once it finished running two files will be created in a folder called `results`:
* out-svd-simple-<timestamp> : Results in .csv format
* out-svd-simple-<timestamp>-params : A copy of the conf.json with the best argument selected

To rerun the experiment, the params file produced can be used directly:
```
./main.py --training-data data_train.csv --submission-data sampleSubmission.csv --model-conf results/out-svd-simple-<timestamp>-params
```

### Available prediction methods:
* `buildEnsemble`
* `SGD` (includes a non negative U and V option)
* `kmeans_simple`
* `kmeans_iterative`
* `svd_iterative`
* `svd_simple`
* `simple_baseline`

Consult `collabfilter.py` on more details on each of those methods

## Reproducing our results

This guide explains how to reproduce all the results in our paper.

### Reproducing our Kaggle score

1) Split the training data
```
./split_training.py --training-data data_train.csv --output-name data_train_ninetenths --fraction 0.9
```
The file ending with '-first' contains 90% of the data. The file ending with '-second' contains the rest.

2) Make sure folders ```additional``` and ```ensemble``` are empty. If those folders do not exist, they will be created automatically.

3) Train the following models(folder `confs` with correct configurations is included):
```
./main.py --training-data data_train_ninetenths-first.csv --submission-data sampleSubmission.csv --model-conf confs/kaggle/svd_iterative_conf.json --store-in-ensemble
```
```
./main.py --training-data data_train_ninetenths-first.csv --submission-data sampleSubmission.csv --model-conf confs/kaggle/svd_simple_conf.json --store-in-ensemble
```
```
./main.py --training-data data_train_ninetenths-first.csv --submission-data sampleSubmission.csv --model-conf confs/kaggle/SGD1_conf.json --store-in-ensemble
```
```
./main.py --training-data data_train_ninetenths-first.csv --submission-data sampleSubmission.csv --model-conf confs/kaggle/SGD2_conf.json --store-in-ensemble
```
```
./main.py --training-data data_train_ninetenths-first.csv --submission-data sampleSubmission.csv --model-conf confs/kaggle/SGD3_conf.json --store-in-ensemble
```
```
./main.py --training-data data_train_ninetenths-first.csv --submission-data sampleSubmission.csv --model-conf confs/kaggle/kmeans_simple_conf.json --store-in-ensemble
```
```
./main.py --training-data data_train_ninetenths-first.csv --submission-data sampleSubmission.csv --model-conf confs/kaggle/kmeans_iterative_conf.json --store-in-ensemble
```

4) Train GBDT:
```
./main.py --training-data data_train_ninetenths-second.csv --submission-data sampleSubmission.csv --model-conf confs/kaggle/GBDT_conf.json
```

5) Final result will be in the `results` folder


### Comparison of different imputation methods

Run the script `test_impute.py` provided:
```
./test_impute.py --training-data data_train.csv --submission-data sampleSubmission.csv --model-conf confs/test_impute_conf.json
```

### SGD Plot - Training error vs. Validation error

Run the following command:
```
./main.py --training-data data_train_ninetenths-first.csv --validation-data data_train_ninetenths-second.csv --submission-data sampleSubmission.csv --model-conf confs/SGD_plot_conf.json
```

## Adding a new collaborative filtering method

In order to add a new prediction method, all it is needed is to add a new function in the file `collabfilter.py` with the correct signature. Once a function is there, it is automatically available to use via `conf.json`

Example:
```
def example_function(baseline_matrix,
                     index_matrix,
                     output_cells,
                     validation_matrix=None,
                     verboseCLI=False,
                     CVMode=False,
                     <additional params added here>):

    <code here>

    if validation_matrix is not None:
        <calculate validation error>
        <apply early stopping if possible>

    if CVMode is True:
        <this code is executed only during crossvalidation>
    if verboseCLI is True:
        print('This is only printed if verbose flag is given')

    <more code here>

    final_complete_matrix = ...  # it is possible to only complete the entries in output_cells instead of the whole matrix
    return final_complete_matrix  # The matrix(only the output_cells) is automatically converted to .csv format later
```

## Authors

* Andreas Georgiou
* Marc Ilunga
* Nico Schottelius
* Sarah Plocher

## Acknowledgments

* We would like to thank all the authors referenced in our paper
