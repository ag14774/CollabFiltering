#!/usr/bin/env python3

"""
Regressors used for ensemble. The format followed is the same as sklearn.
Each class needs to have a function fit(X, y) and a predict(X)
"""

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE


class LGBMRegressorWrapper(LGBMRegressor):
    def fit(self,
            X,
            y,
            sample_weight=None,
            init_score=None,
            eval_set=None,
            eval_names=None,
            eval_sample_weight=None,
            eval_init_score=None,
            eval_metric='l2',
            early_stopping_rounds=None,
            verbose=True,
            feature_name='auto',
            categorical_feature='auto',
            callbacks=None,
            print_importance=True):
        self = super().fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks)
        if print_importance is True:
            print("feature importances: {}".format(self.feature_importances_))
        print()
        return self


class RFELightGBM(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_features_to_select=None,
                 step=1,
                 verboseRFE=0,
                 boosting_type='gbdt',
                 num_leaves=31,
                 max_depth=-1,
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample_for_bin=200000,
                 objective=None,
                 class_weight=None,
                 min_split_gain=0.0,
                 min_child_weight=0.001,
                 min_child_samples=20,
                 subsample=1.0,
                 subsample_freq=1,
                 colsample_bytree=1.0,
                 reg_alpha=0.0,
                 reg_lambda=0.0,
                 random_state=None,
                 n_jobs=-1,
                 silent=True,
                 **kwargs):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verboseRFE = verboseRFE
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.class_weight = class_weight
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.silent = silent
        self.kwargs = kwargs
        self.rfe = None

    def fit(self, X, y=None):
        print("starting recursive feature elimination - target: {} features".
              format(self.n_features_to_select))
        base_estimator = LGBMRegressorWrapper(
            boosting_type=self.boosting_type,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample_for_bin=self.subsample_for_bin,
            objective=self.objective,
            class_weight=self.class_weight,
            min_split_gain=self.min_split_gain,
            min_child_weight=self.min_child_weight,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            silent=self.silent,
            **self.kwargs)
        self.rfe = RFE(
            base_estimator,
            n_features_to_select=self.n_features_to_select,
            step=self.step,
            verbose=self.verboseRFE)
        self.rfe.fit(X, y)
        print()
        print("final feature ranking using RFE: {}".format(self.rfe.ranking_))
        return self

    def predict(self, X, y=None):
        print("predicting data with shape: {}".format(X.shape))
        return self.rfe.predict(X)

    def score(self, X, y):
        return self.rfe.score(X, y)
