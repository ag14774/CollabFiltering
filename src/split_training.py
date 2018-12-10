#!/usr/bin/env python3
"""
Takes the path to the training data and splits it so that the first file
produced contains 'fraction' of the data and the rest goes to the second
Example usage:
python3 split_training.py --training-data ../data/data_train.csv
                          --output-name ../data/data_train_ninetenths
                          --fraction 0.9

This produces the files data_train_ninetenths-first.csv which contains 90%
of the data and data_train_ninetenths-second.csv which contains the remaining.
"""

import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description='Matrix importer for CIL by Nico Schottelius, 2017-2018')
    parser.add_argument(
        '--training-data', help='File to read data from', required=True)
    parser.add_argument(
        '--output-name', help='Name of the first output file', required=True)
    parser.add_argument(
        '--fraction',
        type=float,
        help='Fraction of data in the first output file',
        required=True)

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


if __name__ == '__main__':
    import util
    import numpy as np

    np.random.seed(320)

    parser = get_parser()
    args = parser.parse_args()

    td = util.file_to_valuelist(args.training_data)
    shape = (args.matrix_rows, args.matrix_columns)

    index_matrix = np.full(shape, np.nan)
    util.valuelist_to_matrix(index_matrix, td)

    np.random.shuffle(td)

    lenData1 = round(args.fraction * len(td))
    td1 = util.cells_from_valuelist(td[:lenData1])
    td2 = util.cells_from_valuelist(td[lenData1:])

    util.export_matrix_helper(
        args.output_name + '-first.csv', index_matrix, cells=td1)
    util.export_matrix_helper(
        args.output_name + '-second.csv', index_matrix, cells=td2)
