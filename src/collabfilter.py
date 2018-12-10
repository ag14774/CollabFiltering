#!/usr/bin/env python3
"""
Provides various prediction methods for collaborative filtering and some
helper functions used by those methods.

Note that these functions should not be called directly. They are called
automatically by main.py

Prediction methods have the following signature:

def predFunc(baseline_matrix,
             index_matrix,
             output_cells,
             validation_matrix=None,
             verboseCLI=False,
             CVMode=False,
             **kwargs)

baseline_matrix: Contains a baseline prediction(e.g mean of columns)
                 of *all* cells, including those that are given
index_matrix: Contains the given ratings and NaN in missing entries
output_cells: The cells that need to be predicted.
              The format used is the one given by np.where().
              i.e. ([1,1], [4,5]) means the cells missing are
              (1,4) and (1,5)
validation_matrix: A matrix with validation data a.k.a probe set. Some methods
                   use this for early stopping.
CVMode: This is set to True if the function is running in crossvalidation mode.
"""

import json
import os
import pickle
import sys

import numpy as np

import util
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans


def dot_clip_features(U, V, baseline):
    Ucols = np.split(U, U.shape[1], axis=1)
    Vrows = np.split(V, V.shape[0], axis=0)
    total = baseline.copy()
    for (r, c) in zip(Ucols, Vrows):
        total += np.clip(np.dot(r, c), -4, 4)
    return np.clip(total, 1, 5)


def dot_identity(U, V, baseline):
    return np.dot(U, V) + baseline


def dot_clipped_after(U, V):
    res = np.clip(np.dot(U, V), 1, 5)
    return res


def identity(x):
    return x


def clip(x, bottom, up):
    if x < bottom:
        return bottom
    if x > up:
        return up
    return x


def plotErrors(train_errs, val_errs=None):
    plt.plot(train_errs, label='train error')
    if val_errs is not None and len(val_errs) > 0:
        plt.plot(val_errs, label='probe set error')
    plt.legend()
    plt.show()


def reducedSVD(mat, knew=None, tau=None, returnUV=False):
    """
    Computes the SVD decomposition of mat and modifies the singular values
    in one of the following ways:
    1) Keep only the first knew values
    2) Subtract tau from values setting them to 0 if they become negative.
    The matrix is then recomputed using the new singular values
    """

    if knew is not None and tau is not None:
        raise Exception("knew and tau cannot be set at the same time!")
    if knew is None and tau is None:
        raise Exception("knew and tau cannot be both None!")

    # U:m x k, d:k values, V:k x n
    U, d, V = np.linalg.svd(mat, full_matrices=False)

    if tau is not None:  # SVD Shrinkage
        dnew = d - tau
        dnew = dnew[dnew > 0]
        knew = len(dnew)
    elif knew is not None:  # SVD cutoff
        dnew = d[:knew]

    DDiagonal = np.array(dnew)

    if returnUV is False:
        return dot_clipped_after(U[:, :knew] * DDiagonal, V[:knew, :])
    elif returnUV is True:
        U = U[:, :knew]
        V = V[:knew, :]
        return dot_clipped_after(U * DDiagonal, V), U, V.transpose()


def loadModels(ensembleFolder, resultFolder, verbose=False):
    """
    Loads all matrices from ensembleFolder
    """
    printFile = sys.stdout
    if verbose is False:
        f = open(os.devnull, 'w')
        printFile = f

    ensembleMatrices = []
    ensembleParams = []
    ensembleFilenames = os.listdir(ensembleFolder)
    ensembleFilenames.sort()
    # Load the matrices generated by the other methods
    try:
        print("==================================================")
        print("Loading models...")
        for i, f in enumerate(ensembleFilenames):
            ensembleMatrices.append(np.load(ensembleFolder + f))
            with open(resultFolder + f + '-params', 'r') as file:
                params = json.load(file)
                ensembleParams.append(params)
            print("Model loaded: {}".format(f), file=printFile)
            print("Parameters:", file=printFile)
            print(json.dumps(ensembleParams[-1], indent=2), file=printFile)
            if i != len(ensembleFilenames) - 1:
                print("--------------------------------------", file=printFile)
        print("All models loaded..")
        print("==================================================")
    except EnvironmentError as e:
        print(e)
        sys.exit(1)

    return ensembleMatrices, ensembleParams, ensembleFilenames


# ---------------------------------------------------------------------------------


def buildEnsemble(baseline_matrix,
                  index_matrix,
                  output_cells,
                  validation_matrix=None,
                  verboseCLI=False,
                  CVMode=False,
                  classpath='sklearn.kernel_ridge.KernelRidge',
                  **predictArgs):
    """

    """

    (rows, columns) = index_matrix.shape
    known_inds = np.where(~np.isnan(index_matrix))

    ensembleFolder = 'ensemble/'
    resultFolder = 'results/'
    additionalFolder = 'additional/'

    ensembleMatrices, ensembleParams, ensembleFilenames = loadModels(
        ensembleFolder, resultFolder, verbose=verboseCLI)

    # Create X where rows are known cells and columns are the predictions
    # for that cells(each column coming from a different model)
    # samples x features
    y = index_matrix[known_inds]
    X = np.zeros(
        (len(known_inds[0]), len(ensembleMatrices) + 2), dtype='float32')
    Xout = np.zeros(
        (len(output_cells[0]), len(ensembleMatrices) + 2), dtype='float32')

    # Insert predictions of other models
    for i, mat in enumerate(ensembleMatrices):
        X[:, i] = ensembleMatrices[i][known_inds]
        Xout[:, i] = ensembleMatrices[i][output_cells]

    X = np.clip(X, 1, 5, X)
    Xout = np.clip(Xout, 1, 5, Xout)

    # Insert the user and item id as two additional features
    X[:, len(ensembleMatrices)] = known_inds[0]
    X[:, len(ensembleMatrices) + 1] = known_inds[1]
    Xout[:, len(ensembleMatrices)] = output_cells[0]
    Xout[:, len(ensembleMatrices) + 1] = output_cells[1]

    temp = None
    ensembleMatrices = None

    # We load additional features generated by other models
    # e.g for an entry (i.j) we add the vector u_i and v_j
    # as features
    print("Loading additional features...")
    additionalFilenames = os.listdir(additionalFolder)
    additionalFilenames.sort()
    tuples = []
    # Count features to allocate enough memory
    try:
        counter = 0
        for i, f in enumerate(additionalFilenames):
            with open(additionalFolder + f, 'rb') as file:
                temp = pickle.load(file)
            for tup in temp:
                tuples.append(tup)
                # The models generate two types of features in
                # our 'additional' folder. First, there is a factorisation
                # produced by many of our models here. The second is
                # when we want to add a single feature for the users.
                # e.g the assigned cluster number from K-Means
                # The first element in the tuple determines the type
                # so that it can be handled appropriately.
                if tup[0] == 'user-item-factorisation':
                    U = tup[1]
                    Z = tup[2]
                    counter += U.shape[1]
                    counter += Z.shape[1]
                elif tup[0] == 'user-single-feature':
                    counter += 1
                else:
                    raise Exception("Label not recognized")
    except EnvironmentError as e:
        print(e)
        sys.exit(1)

    # Allocate memory
    X_old = X
    Xout_old = Xout
    X = np.zeros((X_old.shape[0], X_old.shape[1] + counter), dtype='float32')
    X[:, :X_old.shape[1]] = X_old[:, :]
    Xoffset = X_old.shape[1]
    X_old = None
    Xout = np.zeros(
        (Xout_old.shape[0], Xout_old.shape[1] + counter), dtype='float32')
    Xout[:, :Xout_old.shape[1]] = Xout_old[:, :]
    Xoutoffset = Xout_old.shape[1]
    Xout_old = None

    # Insert the loaded features in the new matrix
    for i, tup in enumerate(tuples):
        if tup[0] == 'user-item-factorisation':
            U = tup[1]
            Z = tup[2]
            X[:, Xoffset:Xoffset + U.shape[1]] = U[known_inds[0], :]
            Xoffset += U.shape[1]
            X[:, Xoffset:Xoffset + Z.shape[1]] = Z[known_inds[1], :]
            Xoffset += Z.shape[1]
            Xout[:, Xoutoffset:Xoutoffset + U.shape[1]] = U[output_cells[0], :]
            Xoutoffset += U.shape[1]
            Xout[:, Xoutoffset:Xoutoffset + Z.shape[1]] = Z[output_cells[1], :]
            Xoutoffset += Z.shape[1]
        elif tup[0] == 'user-single-feature':
            feature = tup[1]
            feature = feature.reshape(-1, 1)
            X[:, Xoffset:Xoffset + 1] = feature[known_inds[0], :]
            Xoffset += 1
            Xout[:, Xoutoffset:Xoutoffset + 1] = feature[output_cells[0], :]
            Xoutoffset += 1
    tuples = None
    print("Additional features loaded: {}".format(counter))
    print("==================================================")

    # Get predictor class object from string.
    # It is assumed that there is a fit() and a predict() method.
    classObj = util.get_class(classpath)
    classInst = classObj(**predictArgs)
    print(classInst)
    classInst = classInst.fit(X, y)
    X = None
    y = None
    output = classInst.predict(Xout).reshape(-1)
    Xout = None

    result = np.zeros((rows, columns), dtype='float32')
    result[output_cells] = np.clip(output, 1, 5)

    return result


def initCache(U, Z, baseline_matrix):
    UZ = baseline_matrix + np.dot(U, Z.transpose())
    last_feature = np.full(UZ.shape, U.shape[1] - 1, dtype=np.int)
    # The last column is being modified so do not commit it
    UZ_commited = baseline_matrix + np.dot(U[:, :-1], Z[:, :-1].transpose())
    return (UZ, last_feature, UZ_commited)


def predictRating_identity(r, c, U, Z, cache):
    '''
    cache structure:
    ================
    Assumptions: The only changes we make to U and Z are:
                 * Changing the value of last element of the row
                 * Adding a new element(column of U or Z) to a row
                   but not more than one each time
                 * U and Z always have the same number of columns
    (
        UZ: Stores the current multiplication of U * Z.transpose()
        uz_last_feature: The last feature/column of U and Z used in
                         the multiplication to calculate a particular
                         cell of UZ. One integer for each cell in UZ
        UZ_commited: Stores a "stable" version of UZ calculated using all
                     features except the last(which is currently being updated)
    )
    '''
    (UZ, last_feature, UZ_commited) = cache

    lf = last_feature[r, c]
    UZ[r, c] = UZ_commited[r, c] + U[r, lf] * Z[c, lf]

    # If there is a new column then we commit our current result
    # and start optimising the new column
    if (lf + 1 != U.shape[1]):
        UZ_commited[r, c] = UZ[r, c]
        lf += 1
        last_feature[r, c] = lf
        UZ[r, c] = UZ_commited[r, c] + U[r, lf] * Z[c, lf]

    return UZ[r, c]


def predictRating_clip(r, c, U, Z, cache):
    (UZ, last_feature, UZ_commited) = cache

    lf = last_feature[r, c]
    UZ[r, c] = clip(UZ_commited[r, c] + U[r, lf] * Z[c, lf], 1, 5)

    # A column was added
    if (lf + 1 != U.shape[1]):
        UZ_commited[r, c] += clip(U[r, lf] * Z[c, lf], -4, 4)
        lf += 1
        last_feature[r, c] = lf
        UZ[r, c] = clip(UZ_commited[r, c] + U[r, lf] * Z[c, lf], 1, 5)

    return UZ[r, c]


def trainSingleEpoch(index_matrix,
                     known_inds,
                     U,
                     Z,
                     cache,
                     learning_rate=0.01,
                     l2_coef=0.03,
                     non_negative=False,
                     func=predictRating_identity):
    '''
    Uses index_matrix[row, column] to train the matrices U and Z.
    Because we are using a single sample, this function only trains
    U[d, -1] and Z[d, -1] i.e each time we train the last feature.
    When a feature/column is trained, we add a new column(not in this function)
    '''

    if non_negative is False:
        neginf = -np.inf
    else:
        neginf = 0

    inds = list(range(len(known_inds[0])))
    np.random.shuffle(inds)
    for ind in inds:
        row, column = (known_inds[0][ind], known_inds[1][ind])
        err = index_matrix[row, column] - func(row, column, U, Z, cache)
        Ugradient = -err * Z[column, -1] + l2_coef * U[row, -1]
        Zgradient = -err * U[row, -1] + l2_coef * Z[column, -1]

        U[row, -1] = max(neginf, U[row, -1] - learning_rate * Ugradient)
        Z[column, -1] = max(neginf, Z[column, -1] - learning_rate * Zgradient)


def SGD(baseline_matrix,
        index_matrix,
        output_cells,
        validation_matrix=None,
        verboseCLI=False,
        CVMode=False,
        learning_rate=0.001,
        features=20,
        epochs_per_feat=50,
        l2_coef=0.03,
        function="identity",
        seed=97,
        initial_value=0.1,
        initial_lr_boost_factor=1,
        initial_lr_boost_epochs=0,
        non_negative=False,
        early_stopping=False,
        store_factorisation=False,
        plot_errors=False):

    np.random.seed(seed)

    IV = initial_value

    (rows, columns) = index_matrix.shape
    known_inds = np.where(~np.isnan(index_matrix))
    if validation_matrix is not None:
        val_inds = np.where(~np.isnan(validation_matrix))

    # Identity: Simple SGD with no clipping
    # clip: Clips each feature between -4 and 4 and the final
    # result between 1 and 5
    if function == "identity":
        predictFunc = predictRating_identity
        final_func = dot_identity
    elif function == "clip":
        predictFunc = predictRating_clip
        final_func = dot_clip_features
    else:
        raise Exception("Not supported function for SGD training")

    U = np.full((rows, 1), IV)
    Z = np.full((columns, 1), IV)
    cache = initCache(U, Z, baseline_matrix)

    train_errs = []
    val_errs = []

    for feat in range(1, features + 1):
        early_stop_rounds = 2
        last_val_err = 999999
        for i in range(1, epochs_per_feat + 1):

            if early_stop_rounds == 0 and early_stopping is True:
                print("Validation error is increasing...stopping!!")
                break

            # Increased learning rate for the first iterations
            # to speed up training
            if i <= initial_lr_boost_epochs:
                lr = learning_rate * initial_lr_boost_factor
            else:
                lr = learning_rate

            trainSingleEpoch(
                index_matrix=index_matrix,
                known_inds=known_inds,
                U=U,
                Z=Z,
                cache=cache,
                learning_rate=lr,
                l2_coef=l2_coef,
                non_negative=non_negative,
                func=predictFunc)

            # Calculate training error
            frosquared = np.sum(
                np.square(index_matrix[known_inds] - cache[0][known_inds]))
            err = np.sqrt(frosquared / len(known_inds[0]))
            train_errs.append(err)
            if validation_matrix is None:
                print(("feature: {} - epoch: {} - training error: {}").format(
                    feat, i, err))
            else:
                # Calculate validation error for early stopping
                residuals = [(validation_matrix[row, col] - predictFunc(
                    row, col, U, Z, cache))**2
                             for row, col in zip(val_inds[0], val_inds[1])]
                val_err = np.sqrt(np.sum(residuals) / len(val_inds[0]))
                val_errs.append(val_err)
                print((("feature: {} - epoch: {} - training error: {}"
                        " - validation error: {}")).format(
                            feat, i, err, val_err))

                # We need 3 consecutive epochs with increasing error to stop
                if val_err > last_val_err:
                    early_stop_rounds -= 1
                else:
                    early_stop_rounds = 2
                last_val_err = val_err

        if feat != features:
            U = np.hstack((U, np.full((rows, 1), IV)))
            Z = np.hstack((Z, np.full((columns, 1), IV)))
            print("adding a new column")

    if store_factorisation is True and CVMode is False:
        exporter = util.Exporter()
        exporter.queue_object(('user-item-factorisation', U, Z))

    if plot_errors is True:
        plotErrors(train_errs, val_errs)

    return final_func(U, Z.transpose(), baseline_matrix)


def kmeans_simple(baseline_matrix,
                  index_matrix,
                  output_cells,
                  validation_matrix=None,
                  verboseCLI=False,
                  CVMode=False,
                  random_state=None,
                  n_jobs=1,
                  max_iter=300,
                  batch_size=-1,
                  n_clusters=8,
                  store_assignments=False):

    # matrix is user x items

    matrix = util.merge_matrices(index_matrix, baseline_matrix)
    if batch_size == -1:
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=n_jobs)
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            batch_size=batch_size,
            random_state=random_state)

    print("running kmeans with {} clusters..".format(n_clusters))
    assignments = kmeans.fit_predict(matrix)
    centroids = kmeans.cluster_centers_
    for i in range(n_clusters):
        matrix[assignments == i] = centroids[i]

    if store_assignments is True and CVMode is False:
        exporter = util.Exporter()
        # Export the cluster number of each user
        exporter.queue_object(('user-single-feature', assignments))

    return matrix


def kmeans_iterative(baseline_matrix,
                     index_matrix,
                     output_cells,
                     validation_matrix=None,
                     verboseCLI=False,
                     CVMode=False,
                     random_state=None,
                     n_jobs=1,
                     max_iter=300,
                     batch_size=-1,
                     n_clusters=8,
                     n_rounds=10):

    matrix = util.merge_matrices(index_matrix, baseline_matrix)
    known_inds = np.where(~np.isnan(index_matrix))

    for i in range(1, n_rounds + 1):
        matrix[known_inds] = index_matrix[known_inds]
        if batch_size == -1:
            kmeans = KMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                random_state=random_state,
                n_jobs=n_jobs)
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                batch_size=batch_size,
                random_state=random_state)

        print("round {} - running kmeans with {} clusters..".format(
            i, n_clusters))
        assignments = kmeans.fit_predict(matrix)
        centroids = kmeans.cluster_centers_
        for i in range(n_clusters):
            matrix[assignments == i] = centroids[i]

    return matrix


def svd_iterative(baseline_matrix,
                  index_matrix,
                  output_cells,
                  validation_matrix=None,
                  verboseCLI=False,
                  CVMode=False,
                  knew=None,
                  tau=None,
                  iters=10,
                  store_factorisation=False):
    matrix = util.merge_matrices(index_matrix, baseline_matrix)
    log_frequency = 10  # smaller is more frequent
    known_inds = np.where(~np.isnan(index_matrix))
    for x in range(iters - 1):
        matrix = reducedSVD(matrix, knew=knew, tau=tau)

        if (x % log_frequency == 0):
            # Calculate training error
            err = np.linalg.norm(matrix[known_inds] -
                                 index_matrix[known_inds])**2
            err = err / len(known_inds[0])
            err = np.sqrt(err)
            print("step: {} - error: {}".format(x, err))

        matrix[known_inds] = index_matrix[known_inds]

    result, U, V = reducedSVD(matrix, knew=knew, tau=tau, returnUV=True)

    if store_factorisation is True and CVMode is False:
        exporter = util.Exporter()
        # We only store the first 50 features as those appear to be the
        # most important
        exporter.queue_object(('user-item-factorisation', U[:, :50],
                               V[:, :50]))

    return result


def svd_simple(baseline_matrix,
               index_matrix,
               output_cells,
               validation_matrix=None,
               verboseCLI=False,
               CVMode=False,
               knew=10):
    matrix = util.merge_matrices(index_matrix, baseline_matrix)
    return reducedSVD(matrix, knew=knew)


def simple_baseline(baseline_matrix,
                    index_matrix,
                    output_cells,
                    validation_matrix=None,
                    verboseCLI=False,
                    CVMode=False):
    """Simply return the imputed matrix as prediction"""
    return baseline_matrix
