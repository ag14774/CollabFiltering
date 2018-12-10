#!/usr/bin/env python3
"""
Script to calculate the error of each impute method
using crossvalidation
"""

TESTPARAMS = [{
    "baseline_method": "mean_baseline",
    "baseline_settings": {}
}, {
    "baseline_method": "mean_col_baseline",
    "baseline_settings": {}
}, {
    "baseline_method": "mean_col_with_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 0
    }
}, {
    "baseline_method": "mean_col_with_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 10
    }
}, {
    "baseline_method": "mean_col_with_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 20
    }
}, {
    "baseline_method": "mean_col_with_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 30
    }
}, {
    "baseline_method": "mean_col_with_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 100
    }
}, {
    "baseline_method": "mean_col_with_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 120
    }
}, {
    "baseline_method": "mean_col_with_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 140
    }
}, {
    "baseline_method": "mean_col_with_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 160
    }
}, {
    "baseline_method": "mean_col_with_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 190
    }
}, {
    "baseline_method": "mean_col_with_offset_and_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 0,
        "offset_prior_coef": 10
    }
}, {
    "baseline_method": "mean_col_with_offset_and_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 0,
        "offset_prior_coef": 15
    }
}, {
    "baseline_method": "mean_col_with_offset_and_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 0,
        "offset_prior_coef": 20
    }
}, {
    "baseline_method": "mean_col_with_offset_and_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 15,
        "offset_prior_coef": 15
    }
}, {
    "baseline_method": "mean_col_with_offset_and_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 25,
        "offset_prior_coef": 25
    }
}, {
    "baseline_method": "mean_col_with_offset_and_prior_baseline",
    "baseline_settings": {
        "mean_prior_coef": 50,
        "offset_prior_coef": 25
    }
}]

if __name__ == '__main__':
    import cli
    from CV import CrossValidator
    import collabfilter as CF

    # Get all data from CLI
    everything = cli.get_everything()
    td = everything['td']
    index_matrix = everything['index_matrix']
    conf = everything['conf']
    verboseCLI = everything['verbose']
    shape = index_matrix.shape

    # Get pointer to prediction function using its name
    PREDICTION_FUNCTION = CF.__dict__[conf['predictor']]

    # Run CV
    arglist = conf['param_list']
    round_thresh = conf['rounding_threshold']
    results = []
    for param in TESTPARAMS:
        td2 = td.copy()
        conf['baseline_method'] = param['baseline_method']
        conf['baseline_settings'] = param['baseline_settings']
        cv = CrossValidator(
            func=PREDICTION_FUNCTION,
            shape=shape,
            baseline_method=conf['baseline_method'],
            baseline_settings=conf['baseline_settings'],
            round_thresh=round_thresh,
            k=conf['CV_folds'],
            verboseCLI=verboseCLI)
        cv.generateFolds(td2)
        bestArg, bestScore = cv.selectBestArgs(arglist)
        results.append((conf['baseline_method'], conf['baseline_settings'],
                       bestScore))

    for method, settings, score in results:
        print(method, settings, score)
