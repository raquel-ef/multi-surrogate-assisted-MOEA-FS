# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:11:02 2022

@author: Raquel
"""
from platypus import Problem, Binary
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate 
from statistics import mode
import config.config as config


class Wrapper_ML_Classification(Problem):
    def __init__(self, train_X, train_y, model, nVar=2, nobjs=2, nSplits=3, nRepeats=1):
        super().__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.directions[0] = Problem.MAXIMIZE  # Maximize accuracy
        self.train_X = train_X
        self.train_y = train_y
        self.model = model
        self.nSplits = nSplits
        self.nRepeats = nRepeats

    def evaluate(self, solution):
        mask = [sol[0] for sol in solution.variables]

        # Select columns based on mask
        Dprime = self.train_X.iloc[:, mask] if any(mask) else None
        N = Dprime.shape[1] if Dprime is not None else 0

        rkf = RepeatedStratifiedKFold(n_splits=self.nSplits, n_repeats=self.nRepeats, random_state=config.SEED_VALUE)

        if N == 0:  # No features selected, use mode of y_train as constant prediction
            acc = [
                balanced_accuracy_score(y_test, [0] * len(y_test))
                for train_idx, test_idx in rkf.split(self.train_y, self.train_y)
                for y_train, y_test in [(self.train_y.iloc[train_idx], self.train_y.iloc[test_idx])]
            ]
            scores = {"test_balanced_accuracy": acc}

        else:  # Perform Cross-Validation
            scores = cross_validate(
                self.model, Dprime, self.train_y.values.ravel(),
                cv=rkf, scoring=['balanced_accuracy']
            )

        solution.objectives[:] = [np.mean(scores['test_balanced_accuracy']), N]
        