#!/usr/bin/env python3
"""
Simple implementation of k-fold crossvalidation
"""

import math

import numpy as np

import util


class CrossValidator:
    def __init__(self,
                 func,
                 shape=(10000, 1000),
                 baseline_method="undef",
                 baseline_settings={},
                 round_thresh=0.1,
                 k=5,
                 verboseCLI=False,
                 seed=71):
        self.func = func  # Prediction function to use
        self.shape = shape  # Shape of matrix

        # Method and settings to use when creating the baseline matrix
        self.baseline_method = baseline_method
        self.baseline_settings = baseline_settings

        self.round_thresh = round_thresh  # Round numbers close to an integer
        self.k = k  # Number of folds. Set to 0 to disable CV
        self.seed = seed  # For reproducible experiments
        self.verboseCLI = verboseCLI  # --verbose flag given by CLI
        self.folds = None
        self.output = None
        self.scores = None

    def generateFolds(self, td):
        self.folds = []
        np.random.seed(self.seed)
        np.random.shuffle(td)
        # Create k buckets
        buckets = [td[i::self.k] for i in range(self.k)]
        for i in range(self.k):
            valdata = buckets[i]
            traindata = [x for j, b in enumerate(buckets) for x in b if j != i]
            self.folds.append((traindata, valdata))

    def singleFold(self, td, vd, extra_val_data=None, **kwargs):
        # **kwargs is additional parameters to pass to self.func
        index_matrix = np.full(self.shape, np.nan)
        util.valuelist_to_matrix(index_matrix, td)
        output_cells = util.cells_from_valuelist(vd)
        baseline_matrix = util.baseline_prediction(
            index_matrix,
            method=self.baseline_method,
            indices=None,
            **self.baseline_settings)

        result = self.func(
            baseline_matrix=baseline_matrix,
            index_matrix=index_matrix,
            output_cells=output_cells,
            validation_matrix=extra_val_data,
            verboseCLI=self.verboseCLI,
            CVMode=True,
            **kwargs)

        if self.round_thresh == 0:
            result_rounded = result
        else:
            result_rounded = result.copy()
            result_rounded = np.round(result_rounded)
            diff = np.abs(result_rounded - result)
            result_rounded[diff > self.round_thresh] = result[
                diff > self.round_thresh]

        output = []
        for tup in vd:
            (r, c, v) = tup
            output.append((r, c, result_rounded[r, c]))
        return output

    def allFolds(self, extra_val_data=None, **kwargs):
        print(
            "starting cross-validation with {} folds".format(self.k),
            flush=True)
        self.output = []
        for i, tup in enumerate(self.folds):
            print("running fold number {}".format(i + 1), flush=True)
            (td, vd) = tup
            fold_out = self.singleFold(td, vd, extra_val_data, **kwargs)
            self.output.append(fold_out)

    def score(self):
        total = 0.0
        count = 0
        fold_loss = []  # loss per fold(used for variance calculation)
        for i, fold in enumerate(self.folds):
            (_, vd) = fold
            total_fold = 0.0
            count_fold = 0
            for j, tup in enumerate(vd):
                (r, c, v) = tup
                (rpred, cpred, vpred) = self.output[i][j]
                if rpred != r or cpred != c:
                    raise Exception("This should never happen!")
                total += (vpred - v)**2
                count += 1
                total_fold += (vpred - v)**2
                count_fold += 1
            fold_loss.append(math.sqrt(total_fold / count_fold))
        res = math.sqrt(total / count)
        variance = np.var(fold_loss)
        return res, variance

    def selectBestArgs(self, arglist, validation_matrix=None):
        self.scores = []
        self.variances = []
        if self.k == 0:
            print(
                "skipping CV. argument selected: {}".format(arglist[0]),
                flush=True)
            return arglist[0], -1
        for kwargs in arglist:
            print()
            print("testing arguments: {}".format(kwargs), flush=True)
            self.allFolds(validation_matrix, **kwargs)
            currScore, currVar = self.score()
            self.scores.append(currScore)
            self.variances.append(currVar)
            print("predicted score: {} +/- {}".format(currScore,
                                                      math.sqrt(currVar)))
        bestArgIndex = np.argmin(self.scores)
        bestScore = np.min(self.scores)
        print(
            "best score: {} - best argument selected: {}".format(
                bestScore, arglist[bestArgIndex]),
            flush=True)
        return arglist[bestArgIndex], bestScore
