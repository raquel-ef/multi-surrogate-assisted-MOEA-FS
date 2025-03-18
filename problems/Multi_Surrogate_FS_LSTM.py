# -*- coding: utf-8 -*-
"""
@author: Raquel
"""

from platypus import *
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.models import load_model
import glob
import config.config as config

# Dynamically load all models
model_files = sorted(glob.glob(f'../models/{config.DATASET_SAVE_NAME}-surrogate-LSTM-[0-9]*.h5'))
models = [load_model(f) for f in model_files]


class Multi_Surrogate_FS_LSTM(Problem):
    def __init__(self, X_cv, Y_cv, nVar=2, nobjs=2):
        super(Multi_Surrogate_FS_LSTM, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.X_cv = X_cv
        self.Y_cv = Y_cv

    def evaluate(self, solution):
        mask = np.array([sol[0] for sol in solution.variables], dtype=bool)
        num_selected_features = mask.sum()

        rmse_test = []
        for X, Y, model in zip(self.X_cv, self.Y_cv, models):
            DprimeTest = np.array(X)
            DprimeTest[:, ~mask] = 0.0 

            # Reshape for LSTM input
            DprimeTest_timeseries = DprimeTest.reshape(DprimeTest.shape[0], 1, DprimeTest.shape[1])

            # Compute RMSE
            predTest = model.predict(DprimeTest_timeseries)
            rmse_test.append(root_mean_squared_error(Y, predTest.ravel()))

        solution.objectives[:] = [np.mean(rmse_test), num_selected_features]

