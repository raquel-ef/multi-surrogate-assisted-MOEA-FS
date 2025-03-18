# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:11:02 2022

@author: Raquel
"""
from platypus import Problem, Binary
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_validate 
import config.config as config


class Wrapper_ML(Problem):
    def __init__(self, train_X, train_y, model, nVar=2, nobjs=2, nSplits=5, nRepeats=1):
        super(Wrapper_ML, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.train_X = train_X
        self.train_y = train_y
        self.model = model
        self.nSplits = nSplits
        self.nRepeats = nRepeats
        self.rkf = RepeatedKFold(n_splits=nSplits, n_repeats=nRepeats, random_state=config.SEED_VALUE)

    def evaluate(self, solution):
        mask = np.array([sol[0] for sol in solution.variables], dtype=bool)
        selected_features = self.train_X.loc[:, mask] 
        num_features = selected_features.shape[1]

        if num_features == 0:
            # Compute RMSE using the mean of train_y when no features are selected
            y_array = self.train_y.to_numpy().ravel()
            mean_y_train = np.mean(y_array)

            rmse_scores = [
                -root_mean_squared_error(y_array[test], np.full_like(y_array[test], mean_y_train))
                for _, test in self.rkf.split(y_array)
            ]
        else:
            # Perform cross-validation with the selected features
            rmse_scores = cross_validate(
                self.model, selected_features, self.train_y.to_numpy().ravel(),
                cv=self.rkf, scoring='neg_root_mean_squared_error'
            )['test_score']

        solution.objectives[:] = [-np.mean(rmse_scores), num_features]
        