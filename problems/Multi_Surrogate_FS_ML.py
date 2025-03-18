# -*- coding: utf-8 -*-
"""
@author: Raquel
"""

from platypus import *
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import pickle 
from pathlib import Path


class Multi_Surrogate_FS_ML(Problem):

    def __init__(self, X_cv, Y_cv, regex, nVar=2, nobjs=2):
        super(Multi_Surrogate_FS_ML, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.X_cv = X_cv
        self.Y_cv = Y_cv

        # Load models
        self.models = [pickle.load(open(f, 'rb')) for f in Path().glob(regex) if f.is_file()]

    def evaluate(self, solution):
        mask = np.array([var[0] for var in solution.variables], dtype=bool)
        rmse_list = []

        for X, Y, model in zip(self.X_cv, self.Y_cv, self.models):
            DprimeTest = pd.DataFrame(X).copy()
            DprimeTest.loc[:, ~mask] = 0.0  
            predTest = model.predict(DprimeTest.to_numpy())
            rmse_list.append(root_mean_squared_error(Y, predTest.ravel()))

        solution.objectives[:] = [np.mean(rmse_list), mask.sum()]
