#!/usr/bin/env python3
"""
Functions for baseline prediction(e.g mean of columns)
and other utility functions
"""

import importlib
import json
import os
import pickle
import re
import sys
import time

import numpy as np


class Singleton(type):
    '''
    Reference:
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    Can be used to instantiate any class using the singleton design pattern.
    '''
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class Exporter(metaclass=Singleton):
    """
    Exporter class. It uses the singleton pattern so only one instance will
    be created. Repeated invocations will return the same instance.
    The reason for this is so that objects can be added to a queue to export
    later.
    """

    def __init__(self):
        self.queue = []

    def queue_object(self, obj):
        self.queue.append(obj)

    def export_all(self, output_mat, init_conf, args_used, store_in_ensemble,
                   output_cells):
        """
        Output files created:
        -results/out-<method>-<timestamp>: Output data in Kaggle format
        -results/out-<method>-<timestamp>-params: A JSON configuration file
         that can be reused as is to repeat the exact same experiment. If
         CV was used, then only the selected parameters are included and CV
         will be disabled automatically by setting k=0.
        -ensemble/out-<method>-<timestamp>: This file contains the whole output
         matrix in the numpy format(numpy uses pickle to dump data).
         (OPTIONAL: Only when store_in_ensemble is true)
        -additional/out-<method>-<timestamp>: Any additional data that was
         queued by a call to self.queue_object
        """
        timestamp = round(time.time())
        export_matrix(
            output_mat,
            init_conf,
            args_used,
            store_in_ensemble=store_in_ensemble,
            cells=output_cells,
            timestamp=timestamp)
        if len(self.queue) > 0:
            additionalFolder = "additional/"
            if not os.path.exists(additionalFolder):
                os.makedirs(additionalFolder)
            fname = "out-{}-{}".format(init_conf['predictor'], timestamp)
            with open(additionalFolder + fname, 'wb') as fd:
                pickle.dump(self.queue, fd)


def export_matrix(output_mat,
                  init_conf,
                  args_used,
                  store_in_ensemble=False,
                  cells=None,
                  timestamp=None):

    if timestamp is None:
        timestamp = round(time.time())

    # Disable CV for next time and set chosen arguments
    init_conf['CV_folds'] = 0
    init_conf['param_list'] = [args_used]

    resultFolder = 'results/'
    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)
    ensembleFolder = 'ensemble/'
    if not os.path.exists(ensembleFolder):
        os.makedirs(ensembleFolder)
    fname = "out-{}-{}".format(init_conf['predictor'], timestamp)

    with open(resultFolder + fname + '-params', 'w') as fd:
        json.dump(init_conf, fd, indent=2)

    if store_in_ensemble is True:
        output_mat.dump(ensembleFolder + fname)

    export_matrix_helper(resultFolder + fname, output_mat, cells)


def export_matrix_helper(fname, matrix, cells=None):
    """
    Export matrix:
    - Shift rows/columns by 1, as input is 1 and not 0 based
    - Output format is
    Id,Prediction
    r37_c1,3

    Looping over a matrix goes via rows:
    >>> for x in np.arange(12).reshape(3,4):
    ...     print("-- {} ".format(x))
    ...
    -- [0 1 2 3]
    -- [4 5 6 7]
    -- [ 8  9 10 11]

    If cells is not None, only export given cells.
    cells must be iteratable of tuple

    """

    with open(fname, "w") as fd:
        fd.write("Id,Prediction\n")
        if cells:
            for r, c in zip(cells[0], cells[1]):
                fd.write("{}\n".format(cell_to_outformat(matrix, (r, c))))
        else:
            row_idx = 0
            for row in matrix:
                col_idx = 0

                for col in matrix[row_idx]:
                    fd.write("{}\n".format(
                        cell_to_outformat(matrix, (row_idx, col_idx))))
                    col_idx += 1

                row_idx += 1


def cell_to_outformat(matrix, cell):
    row = cell[0] + 1
    col = cell[1] + 1
    val = matrix[cell[0], cell[1]]

    return "r{}_c{},{}".format(row, col, val)


def lines_to_valuelist(lines):
    """Given lines from the specified format, return a
    list of tuples (row, col, value)
    """

    res = []

    for line in lines:
        parsed_line = re.search(
            "r(?P<row>[0-9]*)_c(?P<column>[0-9]*),(?P<value>.*)", line)

        row = int(parsed_line.group('row')) - 1
        column = int(parsed_line.group('column')) - 1
        value = float(parsed_line.group('value'))

        res.append((row, column, value))

    return res


def file_to_valuelist(fname):
    try:
        with open(fname, "r") as f:
            lines = f.readlines()
            lines = lines[1:]  # Skip header
            lines = [line.rstrip('\n') for line in lines]  # Remove silly \n

    except Exception as e:
        print(e)
        sys.exit(1)

    return lines_to_valuelist(lines)


def valuelist_to_matrix(matrix, valuelist):
    for row, col, val in valuelist:
        matrix[row, col] = val

    return matrix


def cells_from_valuelist(valuelist):
    maxr = max(valuelist, key=lambda item: item[0])[0]
    maxc = max(valuelist, key=lambda item: item[1])[1]
    temp_matrix = np.full((maxr + 1, maxc + 1), np.nan)
    valuelist_to_matrix(temp_matrix, valuelist)
    return np.where(~np.isnan(temp_matrix))


def values_from_valuelist(valuelist):
    return [x[2] for x in valuelist]


def get_small_random_matrix():
    cols = 5
    rows = 10

    matrix = np.random.randint(5 + 1, size=(rows, cols))

    return matrix


def get_class(string):
    """
    Used to dynamically load a class given the name of the class
    e.g get_class(sklearn.neural_network.MLPClassifier) will return
    the actual class object from sklearn.
    """
    k = string.rfind('.')
    if k != -1:
        moduleStr = string[:k]
        funcStr = string[k + 1:]
        module = importlib.import_module(moduleStr)
        return module.__dict__[funcStr]
    else:
        return globals()[string]


# ===================== baseline prediction functions =====================


def mean_baseline(matrix, indices):
    rows, cols = matrix.shape
    mean = np.nanmean(matrix)
    matrix = np.full((rows, cols), mean)
    return matrix


def mean_col_baseline(matrix, indices):
    rows, cols = matrix.shape
    col_mean = np.nanmean(matrix, axis=0)
    matrix = np.tile(col_mean, (rows, 1))
    return matrix


def mean_col_with_prior_baseline(matrix, indices, mean_prior_coef=190):
    rows, cols = matrix.shape
    K = mean_prior_coef
    col_sum = np.nansum(matrix, axis=0)
    col_count = np.count_nonzero(~np.isnan(matrix), axis=0)
    global_mean = np.nanmean(matrix)
    posterior_col_mean = (K * global_mean + col_sum) / (K + col_count)
    matrix = np.tile(posterior_col_mean, (rows, 1))
    return matrix


def mean_col_with_offset_and_prior_baseline(matrix,
                                            indices,
                                            mean_prior_coef=50,
                                            offset_prior_coef=25):
    rows, cols = matrix.shape
    # average item rating
    K = mean_prior_coef
    col_sum = np.nansum(matrix, axis=0)
    col_count = np.count_nonzero(~np.isnan(matrix), axis=0)
    global_mean = np.nanmean(matrix)
    posterior_col_mean = (K * global_mean + col_sum) / (K + col_count)

    K2 = offset_prior_coef
    offset_matrix = matrix - posterior_col_mean
    # average offset of user. e.g. how far away from the
    # average does this user rate (on average)
    offset_sum = np.nansum(offset_matrix, axis=1)
    offset_count = np.count_nonzero(~np.isnan(offset_matrix), axis=1)
    offset_avg = (K2 * 0 + offset_sum) / (K2 + offset_count)

    # tile averages to create a 10000 x 1000 matrix
    col_mean_tiled = np.tile(posterior_col_mean, (rows, 1))

    # tile averages to create a 10000 x 1000 matrix
    offset_avg_tiled = np.tile(offset_avg, (cols, 1)).transpose()

    matrix = col_mean_tiled + offset_avg_tiled
    return matrix


def null_baseline(matrix, indices):
    rows, cols = matrix.shape
    matrix = np.zeros((rows, cols))
    return matrix


def undef_baseline(matrix, indices):
    return matrix


def baseline_prediction(index_matrix,
                        method="undef_baseline",
                        indices=None,
                        **kwargs):
    '''
    Takes a sparse matrix as input and uses a 'method' to make
    a baseline prediction for all cells(including the non-empty ones)
    '''

    if indices is None:
        indices = np.where(np.isnan(index_matrix))

    matrix = index_matrix.copy()
    matrix[indices] = np.nan

    try:
        matrix = globals()[method](matrix, indices, **kwargs)
    except Exception:
        raise Exception("No function named {} found.".format(method))

    return matrix


def merge_matrices(index_matrix, baseline_matrix, indices=None):
    '''
    Takes the baseline predictions from the baseline_matrix and inserts them
    to the unknown entries(NaNs) in index_matrix. The known entries remain
    untouched.
    '''

    if indices is None:
        indices = np.where(np.isnan(index_matrix))

    filled_matrix = index_matrix.copy()
    filled_matrix[indices] = np.nan
    filled_matrix[indices] = baseline_matrix[indices]

    return filled_matrix


# =======================================================================
