#!/usr/bin/env python3
"""
Provides the functions required to read and parse command line arguments
and converts input data to numpy arrays.
"""

import argparse
import json

import numpy as np

import util


def get_parser():
    parser = argparse.ArgumentParser(
        description='Collaborative Filtering software for CIL')
    parser.add_argument(
        '--training-data',
        help='File to read training data from',
        required=True)
    parser.add_argument(
        '--submission-data',
        help='File to read submission data from',
        required=True)
    parser.add_argument(
        '--validation-data', help='File to read validation data from')
    parser.add_argument(
        '--model-conf',
        help='File to read the model configuration',
        required=True)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument(
        '--store-in-ensemble',
        action='store_true',
        help=('Set this to true if you want to store the '
              'total matrix for later use in ensemble'))
    parser.add_argument(
        '--no-output', action='store_true', help='Disable all output')

    parser.add_argument(
        '--matrix-rows',
        type=int,
        help='Rows for the matrix',
        required=False,
        default=10000)
    parser.add_argument(
        '--matrix-columns',
        type=int,
        help='Columns for the matrix',
        required=False,
        default=1000)

    return parser


def get_shape(args):
    return (args.matrix_rows, args.matrix_columns)


def cli_to_valuelist(args):
    """Reads input data"""

    train_valuelist = util.file_to_valuelist(args.training_data)
    submission_valuelist = util.file_to_valuelist(args.submission_data)
    validation_valuelist = None
    if args.validation_data is not None:
        validation_valuelist = util.file_to_valuelist(args.validation_data)

    return (train_valuelist, submission_valuelist, validation_valuelist)


def read_conf(filename):
    """Reads JSON config file"""

    with open(filename, 'r') as fd:
        conf = json.load(fd)
    return conf


def get_everything():
    """
    Do everything needed to lift the heavy CLI work from the
    math part

    Returns:
    - training data as a list
    - baseline prediction specified in the model config
    - training data as a matrix with NaN when entry is missing
    - validation data used for early stopping by some methods(OPTIONAL)
    - list of entries to print/predict
    - Model configuration file in JSON format
    - verbose flag
    - store_in_ensemble flag
    - no_output flag used for debugging
    """

    # Put everything into a dict - easy to extend
    everything = {}

    parser = get_parser()
    args = parser.parse_args()

    conf = read_conf(args.model_conf)
    td, sd, vd = cli_to_valuelist(args)
    shape = get_shape(args)
    cells_to_print = util.cells_from_valuelist(sd)

    # Matrix with NaN + known values
    index_matrix = np.full(shape, np.nan)
    util.valuelist_to_matrix(index_matrix, td)

    validation_matrix = None
    if vd is not None:
        validation_matrix = np.full(shape, np.nan)
        util.valuelist_to_matrix(validation_matrix, vd)

    # Matrix + missing indices
    baseline_matrix = util.baseline_prediction(
        index_matrix,
        method=conf['baseline_method'],
        indices=None,
        **conf['baseline_settings'])

    everything['td'] = td
    everything['baseline_matrix'] = baseline_matrix
    everything['index_matrix'] = index_matrix
    everything['validation_matrix'] = validation_matrix
    everything['cells_to_print'] = cells_to_print
    everything['conf'] = conf
    everything['verbose'] = args.verbose
    everything['store_in_ensemble'] = args.store_in_ensemble
    everything['no_output'] = args.no_output

    # Have fun!
    return everything


if __name__ == '__main__':
    print("""
Look at nico.py and copy the parser code to yourname.py
""")
