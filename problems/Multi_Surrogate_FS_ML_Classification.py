# -*- coding: utf-8 -*-
"""
@author: Raquel
"""

from platypus import Problem, Binary
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import pickle 
from pathlib import Path


class Multi_Surrogate_FS_ML(Problem):

    def __init__(self, X_cv, Y_cv, regex, nVar=2, nobjs=2):
        super(Multi_Surrogate_FS_ML, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.directions[0] = Problem.MAXIMIZE  # Maximize accuracy
        self.X_cv = X_cv
        self.Y_cv = Y_cv

        # Load models
        self.models = [pickle.load(open(f, 'rb')) for f in Path().glob(regex) if f.is_file()]

    def evaluate(self, solution):
        mask = np.array([var[0] for var in solution.variables], dtype=bool)
        acc_list = []

        for X, Y, model in zip(self.X_cv, self.Y_cv, self.models):
            DprimeTest = pd.DataFrame(X).copy()
            DprimeTest.loc[:, ~mask] = 0.0  
            predTest = model.predict(DprimeTest.to_numpy())
            acc_list.append(balanced_accuracy_score(Y, predTest.ravel()))

        solution.objectives[:] = [np.mean(acc_list), mask.sum()]

        