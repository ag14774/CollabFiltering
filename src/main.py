#!/usr/bin/env python3

"""
Main entry point.

More info on how to run in README
"""


if __name__ == '__main__':
    import numpy as np
    import cli
    import util
    from CV import CrossValidator
    import collabfilter as CF

    # Get all data from CLI
    everything = cli.get_everything()
    td = everything['td']
    baseline_matrix = everything['baseline_matrix']
    index_matrix = everything['index_matrix']
    validation_matrix = everything['validation_matrix']
    cells_to_print = everything['cells_to_print']
    conf = everything['conf']
    verboseCLI = everything['verbose']
    store_in_ensemble = everything['store_in_ensemble']
    no_out = everything['no_output']
    shape = index_matrix.shape

    # Get pointer to prediction function using its name
    PREDICTION_FUNCTION = CF.__dict__[conf['predictor']]

    # Run CV
    arglist = conf['param_list']
    round_thresh = conf['rounding_threshold']
    cv = CrossValidator(
        func=PREDICTION_FUNCTION,
        shape=shape,
        baseline_method=conf['baseline_method'],
        baseline_settings=conf['baseline_settings'],
        round_thresh=round_thresh,
        k=conf['CV_folds'],
        verboseCLI=verboseCLI)
    cv.generateFolds(td)
    bestArg, bestScore = cv.selectBestArgs(arglist, validation_matrix)

    # Final prediction using the bestArg selected during CV
    # If validation_matrix is provided then early stopping is enabled when
    # possible
    output_mat = PREDICTION_FUNCTION(
        baseline_matrix=baseline_matrix,
        index_matrix=index_matrix,
        output_cells=cells_to_print,
        validation_matrix=validation_matrix,
        verboseCLI=verboseCLI,
        CVMode=False,
        **bestArg)

    # Optional rounding to closest integer
    if round_thresh == 0:
        output_rounded = output_mat
    else:
        output_rounded = output_mat.copy()
        output_rounded = np.round(output_rounded)
        diff = np.abs(output_rounded - output_mat)
        output_rounded[diff > round_thresh] = output_mat[diff > round_thresh]

    if no_out is False:
        # Export everything is correct format
        exporter = util.Exporter()
        exporter.export_all(output_rounded, conf, bestArg, store_in_ensemble,
                            cells_to_print)
