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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, InputLayer
from scikeras.wrappers import KerasRegressor 
import random
import config.config as config


class Wrapper_LSTM(Problem):
    def __init__(self, train_X, train_y, nVar=2, nobjs=2, nSplits=5, nRepeats=1):
        super(Wrapper_LSTM, self).__init__(nVar, nobjs)
        self.types[:] = Binary(1)
        self.train_X = train_X
        self.train_y = train_y
        self.nSplits = nSplits
        self.nRepeats = nRepeats
        self.rkf = RepeatedKFold(n_splits=nSplits, n_repeats=nRepeats, random_state=config.SEED_VALUE)

    def lstm_model(self, input_shape):
        tf.random.set_seed(config.SEED_VALUE)
        model = Sequential([
            InputLayer(shape=input_shape),
            LSTM(units=config.N_NEURONS, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
            Dropout(0.2),
            Dense(1, activation="linear")
        ])
        model.compile(loss='mse', optimizer='adam')
        return model

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
            # Reshape input for LSTM
            Dprime_timeseries = selected_features.to_numpy().reshape(selected_features.shape[0], 1, num_features)

            # Create Keras Regressor model
            modelDL = KerasRegressor(build_fn=lambda: self.lstm_model((1, num_features)), 
                                     epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=0)

            # Perform cross-validation
            tf.random.set_seed(config.SEED_VALUE)
            rmse_scores = cross_validate(
                modelDL, Dprime_timeseries, self.train_y.to_numpy().ravel(),
                cv=self.rkf, scoring='neg_root_mean_squared_error', n_jobs=-1
            )['test_score']

        solution.objectives[:] = [-np.mean(rmse_scores), num_features]
        