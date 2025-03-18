import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score, roc_auc_score
from platypus import unique, nondominated
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
import random
from sklearn import preprocessing

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, InputLayer

import config.config as config
from src.data_processing import reshape_timeseries


def calculate_errors(test_y, pred, is_classification=False):
    """
    Calculate error metrics based on the problem type.
    
    Parameters:
    test_y (array-like): True target values.
    pred (array-like): Predicted values.
    is_classification (bool): Whether the problem is classification.
    
    Returns:
    tuple: Metrics based on problem type.
    """
    if is_classification:
        balanced_acc = balanced_accuracy_score(test_y, np.round(pred))

        lb = preprocessing.LabelBinarizer()
        lb.fit(test_y)
        test_y = lb.transform(test_y)
        pred = lb.transform(pred)

        auc_roc = roc_auc_score(test_y, pred, multi_class='ovo', average='weighted')

        return balanced_acc, auc_roc
    
    else:
        mae = mean_absolute_error(test_y, pred)
        rmse = np.sqrt(mean_squared_error(test_y, pred))
        cc = np.corrcoef(test_y.T, pred.T)[1, 0]    
        return mae, rmse, cc


def normalized_means(result_steps, sizes, maximize=False):
    """
    Normalize error metrics from h-steps-ahead predictions to allow model comparison.
    
    Parameters:
    result_steps (list of arrays): Error metrics for h-steps-ahead predictions.
    sizes (list of int): List of sizes used for splitting the data.
    maximize (bool, optional): If True, inverts the metric (for correlation coefficient normalization).
    
    Returns:
    list: Normalized mean values for each split section.
    """
    ms = []
    minim = 0
    maxim = 1

    # Concatenate the error metrics into a single array, using list accumulation
    for data in result_steps:
        ms.append(data if not maximize else 1 - data)  # Avoid double concatenation
    ms = np.concatenate(ms)

    # Normalize the array
    norm = (ms - minim) / (maxim - minim)

    # Split and calculate means more efficiently
    norm = np.array_split(norm, np.cumsum(sizes)[:-1])  # Efficient splitting based on cumulative sizes
    meanList = [np.round(np.mean(i), 6) for i in norm]

    return meanList


def calculate_H_train_test(df, n_steps, is_classification=False):
    """
    Calculate a summary metric "H" based on normalized error metrics.
    Uses RMSE, MAE, and CC for regression, and Balanced Accuracy and AUC ROC for classification.
    
    Parameters:
    df (DataFrame): Data containing metric values for different steps ahead.
    n_steps (int): Number of prediction steps.
    is_classification (bool): Whether the problem is classification or regression.
    
    Returns:
    Tuple (Htrain, Htest): Normalized mean metrics for training and testing.
    """
    def compute_H(step_metrics):
        """ Helper function to compute normalized means and H values. """
        if is_classification:
            stepsBA = np.array([x[1:] for x in step_metrics['ACC BAL']])
            stepsAUC = np.array([x[1:] for x in step_metrics['AUC']])
            
            return pd.DataFrame({
                'BAnorm': normalized_means(stepsBA, [n_steps] * len(stepsBA), maximize=True),
                'AUCnorm': normalized_means(stepsAUC, [n_steps] * len(stepsAUC), maximize=True)
            }).mean(axis=1)
        else:
            stepsRMSE = np.array([x[1:] for x in step_metrics['RMSE']])
            stepsMAE = np.array([x[1:] for x in step_metrics['MAE']])
            stepsCC = np.array([x[1:] for x in step_metrics['CC']])
            
            return pd.DataFrame({
                'RMSEnorm': normalized_means(stepsRMSE, [n_steps] * len(stepsRMSE)),
                'MAEnorm': normalized_means(stepsMAE, [n_steps] * len(stepsMAE)),
                'CCnorm': normalized_means(stepsCC, [n_steps] * len(stepsCC), maximize=True)
            }).mean(axis=1)
    
    metrics_train = {
        'ACC BAL': df['Train ACC BAL StepsAhead'] if is_classification else None,
        'AUC': df['Train AUC StepsAhead'] if is_classification else None,
        'RMSE': df['Train RMSE StepsAhead'] if not is_classification else None,
        'MAE': df['Train MAE StepsAhead'] if not is_classification else None,
        'CC': df['Train CC StepsAhead'] if not is_classification else None
    }
    
    metrics_test = {
        'ACC BAL': df['Test ACC BAL StepsAhead'] if is_classification else None,
        'AUC': df['Test AUC StepsAhead'] if is_classification else None,
        'RMSE': df['Test RMSE StepsAhead'] if not is_classification else None,
        'MAE': df['Test MAE StepsAhead'] if not is_classification else None,
        'CC': df['Test CC StepsAhead'] if not is_classification else None
    }
    
    Htrain = compute_H(metrics_train)
    Htest = compute_H(metrics_test)
    
    return Htrain, Htest


def calculate_H_CV(df, n_steps, is_classification=False):
    """
    Calculate a summary metric "H" based on normalized error metrics.
    Uses RMSE, MAE, and CC for regression, and Balanced Accuracy and AUC ROC for classification.
    
    Parameters:
    df (DataFrame): Data containing metric values for different steps ahead.
    n_steps (int): Number of prediction steps.
    is_classification (bool): Whether the problem is classification or regression.
    
    Returns:
    DataFrame: The original DataFrame with additional normalized metrics and H score.
    """
    
    if is_classification:
        dfSteps = df.loc[df['Mean ACC BAL StepsAhead'].str.len() > 0].copy()

        stepsBA = dfSteps['Mean ACC BAL StepsAhead'].apply(lambda x: x[1:])
        stepsAUC = dfSteps['Mean AUC StepsAhead'].apply(lambda x: x[1:])
        
        ba_norm = normalized_means(stepsBA.tolist(), [n_steps] * len(stepsBA), maximize=True)
        auc_norm = normalized_means(stepsAUC.tolist(), [n_steps] * len(stepsAUC), maximize=True)
        
        normalizedMean = pd.DataFrame({'BAnorm': ba_norm, 'AUCnorm': auc_norm}, index=dfSteps.index)
    else:
        dfSteps = df.loc[df['Mean RMSE StepsAhead'].str.len() > 0].copy()

        stepsRMSE = dfSteps['Mean RMSE StepsAhead'].apply(lambda x: x[1:])
        stepsMAE = dfSteps['Mean MAE StepsAhead'].apply(lambda x: x[1:])
        stepsCC = dfSteps['Mean CC StepsAhead'].apply(lambda x: x[1:])
        
        rmse = normalized_means(stepsRMSE.tolist(), [n_steps] * len(stepsRMSE))
        mae = normalized_means(stepsMAE.tolist(), [n_steps] * len(stepsMAE))
        cc = normalized_means(stepsCC.tolist(), [n_steps] * len(stepsCC), maximize=True)
        
        normalizedMean = pd.DataFrame({'RMSEnorm': rmse, 'MAEnorm': mae, 'CCnorm': cc}, index=dfSteps.index)
    
    normalizedMean['H CV'] = normalizedMean.mean(axis=1)
    return pd.concat([dfSteps, normalizedMean], axis=1)



def predictions_h_stepsahead(testX, testy, model, n_steps, is_classification=False):
    """
    Perform h-steps-ahead predictions using a machine learning model for both classification and regression.
    
    Parameters:
    testX (DataFrame): Feature set for predictions.
    testy (DataFrame): True target values.
    model (Model): Trained machine learning model.
    n_steps (int): Number of steps ahead for prediction.
    is_classification (bool): Whether the problem is classification.
    
    Returns:
    tuple:
        - DataFrame: Error metrics for each prediction step.
        - DataFrame: Predictions for each step ahead.
        - DataFrame: Updated testX with lagged predictions.
    """
    # Extract lags from column names
    predicted_attribute = testy.columns[0]
    lag_columns = testX.filter(regex=(f"{predicted_attribute}.*")).columns
    selected_lags = [int(col.split("_")[2]) for col in lag_columns]

    # Reset indices for test data
    test_X = testX.reset_index(drop=True)
    test_y = testy.reset_index(drop=True)

    # Initialize results DataFrame
    predictions = pd.DataFrame(index=test_X.index)
    results = []

    # 1-step ahead prediction
    predictions["pred1"] = model.predict(test_X).ravel()
    metrics = calculate_errors(test_y, predictions[["pred1"]], is_classification)
    
    if is_classification:
        results.append({'Balanced Accuracy': metrics[0], 'AUC-ROC': metrics[1]})
    else:
        results.append({'RMSE': metrics[1], 'MAE': metrics[0], 'CC': metrics[2]})

    # Handle case when no lagged variables exist
    if not selected_lags:
        print("No lagged variables found")

    else:
        # Multi-step ahead predictions (when lagged variables exist)
        for step in range(2, n_steps + 2):
            for lag in range(1, step):
                shift_lag = step - lag
                if shift_lag == 1 and 1 not in selected_lags:
                    # Replace first lag with the smallest available lag
                    col_name = f"Lag_{predicted_attribute}_{selected_lags[0]}"
                    test_X[col_name] = predictions[f'pred{lag}'].shift(shift_lag)
                elif shift_lag in selected_lags:
                    col_name = f"Lag_{predicted_attribute}_{shift_lag}"
                    test_X[col_name] = predictions[f'pred{lag}'].shift(shift_lag)

            # Drop NaN values before prediction
            valid_X = test_X.dropna()
            pred = model.predict(valid_X.to_numpy()).ravel()

            # Insert NaN padding for alignment
            predictions[f'pred{step}'] = np.concatenate((np.full(step - 1, np.nan), pred))

            # Calculate errors
            valid_preds = predictions[f'pred{step}'][step - 1:]
            metrics = calculate_errors(test_y.iloc[step - 1:], valid_preds.to_frame(), is_classification)
            
            if is_classification:
                results.append({'Balanced Accuracy': metrics[0], 'AUC-ROC': metrics[1]})
            else:
                results.append({'RMSE': metrics[1], 'MAE': metrics[0], 'CC': metrics[2]})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, predictions, test_X


def predictions_h_stepsahead_LSTM(testX, testy, model, n_steps):
    """
    Perform h-steps-ahead predictions using an LSTM model.
    
    Parameters:
    testX (DataFrame): Feature set for predictions.
    testy (DataFrame): True target values.
    model (LSTM Model): Trained LSTM model.
    n_steps (int): Number of steps ahead for prediction.
    
    Returns:
    tuple:
        - DataFrame: Error metrics (RMSE, MAE, CC) for each prediction step.
        - DataFrame: Predictions for each step ahead.
        - DataFrame: Updated testX with lagged predictions.
    """

    test_X = testX.reset_index(drop=True)
    test_y = testy.reset_index(drop=True)
    predicted_attr = test_y.columns[0]
    
    # Extract lags from column names
    listaAtribSelected = sorted([int(col.split("_")[2]) for col in test_X.filter(regex=(predicted_attr + ".*")).columns])

    # Initialize results DataFrame
    predicciones = pd.DataFrame(index=test_X.index)
    dfResultados = []

    if listaAtribSelected:  # Ensure there are lagged target variables for step-ahead predictions
        # 1-step ahead prediction
        x_reshape = test_X.to_numpy().reshape(test_X.shape[0], 1, test_X.shape[1])
        predicciones["pred1"] = model.predict(x_reshape, verbose=0).ravel()

        rmse, mae, cc = calculate_errors(test_y, predicciones[['pred1']])
        dfResultados.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

        # h-step ahead predictions
        for i in range(2, n_steps + 2):
            for j in range(1, i):
                lag = i - j
                if lag == 1 and 1 not in listaAtribSelected:
                    first_lag = listaAtribSelected[0]
                    test_X[f"Lag_{predicted_attr}_{first_lag}"] = predicciones[f'pred{j}'].shift(lag)
                elif lag in listaAtribSelected:
                    test_X[f"Lag_{predicted_attr}_{lag}"] = predicciones[f'pred{j}'].shift(lag)

            arrayX = test_X.dropna().to_numpy().reshape(-1, 1, test_X.shape[1])
            predNa = np.insert(model.predict(arrayX, verbose=0), 0, [np.nan] * (i - 1))

            predicciones[f'pred{i}'] = predNa[:len(predNa)]

            rmse, mae, cc = calculate_errors(test_y.iloc[(i-1):], predicciones[[f'pred{i}']].iloc[(i-1):])
            dfResultados.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

    return pd.DataFrame(dfResultados), predicciones, test_X
    

def train_evaluate_ML(Tx, Ty, results, modelFunction, n_seed, n_generations, n_splits=5, n_repeats=1, colNames=None, is_classification=False):
    """
    Train and evaluate a ML model using repeated k-fold cross-validation and store results,
    supporting both regression and classification.
    
    Parameters:
    Tx (DataFrame): Feature matrix for training.
    Ty (Series): Target variable for training.
    results (list): List of solutions from a multi-objective evolutionary algorithm.
    modelFunction (callable): Function that returns an ML model.
    n_seed (int): Seed value for reproducibility.
    n_generations (int): Number of generations used in optimization.
    n_splits (int): Number of splits for k-fold cross-validation. Default is 5.
    n_repeats (int): Number of repeats for repeated k-fold cross-validation. Default is 1.
    colNames (list): Names of the columns
    is_classification (bool): Whether the problem is classification.
    
    Returns:
    DataFrame: Summary of performance metrics for each selected solution.
    """
    if colNames is None:
        colNames = ['Run', 'Generations', 'RMSE MOEA', 'N', 'RMSE CV', 'MAE CV', 'CC CV', 
                    'Mean RMSE CV', 'Mean MAE CV', 'Mean CC CV', 
                    'RMSE StepsAhead', 'MAE StepsAhead', 'CC StepsAhead', 
                    'Mean RMSE StepsAhead', 'Mean MAE StepsAhead', 'Mean CC StepsAhead',
                    'SelectedAttrib']
    
    dfSolutions = pd.DataFrame(columns=colNames)
    
    # Choose the appropriate cross-validation strategy
    if is_classification:
        cv_strategy = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=n_seed)
    else:
        cv_strategy = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=n_seed)
    
    for sol in unique(nondominated(results)):
        select = [var[0] for var in sol.variables]
        num_selected = int(sol.objectives[1])

        metric1_CV, metric2_CV, metric3_CV = [], [], []
        stepsMetric1CV, stepsMetric2CV, stepsMetric3CV = [], [], []
        
        if num_selected > 0:  
            Tx_selected = Tx.loc[:, select].copy()
            
            for train_index, test_index in cv_strategy.split(Tx_selected, Ty if is_classification else None):
                X_train, X_test = Tx_selected.iloc[train_index], Tx_selected.iloc[test_index]
                y_train, y_test = Ty.iloc[train_index], Ty.iloc[test_index]

                model = modelFunction.fit(X_train, np.array(y_train).ravel())
                pred = model.predict(X_test)
                metrics = calculate_errors(y_test, pred, is_classification)
                
                metric1_CV.append(metrics[0])
                metric2_CV.append(metrics[1])
                metric3_CV.append(metrics[2] if not is_classification else np.nan)
                
                dfSteps, _, _ = predictions_h_stepsahead(X_test, y_test, model, config.N_STEPS, is_classification)

                stepsMetric1CV.append(dfSteps.iloc[:, 0])
                stepsMetric2CV.append(dfSteps.iloc[:, 1])
                stepsMetric3CV.append(dfSteps.iloc[:, 2] if not is_classification else np.nan)
        
        entry = {
            colNames[0]: n_seed,
            colNames[1]: n_generations,
            colNames[2]: sol.objectives[0],
            colNames[3]: num_selected,
            colNames[4]: np.round(metric1_CV, 6),
            colNames[5]: np.round(metric2_CV, 6),
            colNames[6]: np.round(metric3_CV, 6),
            colNames[7]: np.round(np.mean(metric1_CV), 6),
            colNames[8]: np.round(np.mean(metric2_CV), 6),
            colNames[9]: np.round(np.mean(metric3_CV), 6),
            colNames[10]: np.round(stepsMetric1CV, 6),
            colNames[11]: np.round(stepsMetric2CV, 6),
            colNames[12]: np.round(stepsMetric3CV, 6),
            colNames[13]: np.round(np.mean(stepsMetric1CV, axis=0), 6),
            colNames[14]: np.round(np.mean(stepsMetric2CV, axis=0), 6),
            colNames[15]: np.round(np.mean(stepsMetric3CV, axis=0), 6),
            colNames[16]: Tx.columns[select].to_numpy()
        }
        
        dfSolutions = pd.concat([dfSolutions, pd.DataFrame([entry])], ignore_index=True)
        dfSolutions = dfSolutions.dropna(axis=1, how='all')
    
    return dfSolutions




def train_evaluate_LSTM(Tx, Ty, results, modelFunction, n_seed, n_generations, n_splits=5, n_repeats=1):
    """
    Train and evaluate an LSTM model using repeated k-fold cross-validation and store results.
    
    Parameters:
    Tx (DataFrame): Feature matrix for training.
    Ty (Series): Target variable for training.
    results (list): List of solutions from a multi-objective evolutionary algorithm.
    modelFunction (callable): Function that returns an LSTM model.
    n_seed (int): Seed value for reproducibility.
    n_generations (int): Number of generations used in optimization.
    n_splits (int): Number of splits for k-fold cross-validation. Default is 5.
    n_repeats (int): Number of repeats for repeated k-fold cross-validation. Default is 1.
    
    Returns:
    DataFrame: Summary of performance metrics for each selected solution.
    """
    colNames = ['Run', 'Generations', 'RMSE MOEA', 'N', 'RMSE CV', 'MAE CV', 'CC CV', 
                'Mean RMSE CV', 'Mean MAE CV', 'Mean CC CV', 
                'RMSE StepsAhead', 'MAE StepsAhead', 'CC StepsAhead', 
                'Mean RMSE StepsAhead', 'Mean MAE StepsAhead', 'Mean CC StepsAhead',
                'SelectedAttrib']
    
    dfSolutions = pd.DataFrame(columns=colNames)
    
    for sol in unique(nondominated(results)):
        select = [var[0] for var in sol.variables]
        num_selected = int(sol.objectives[1])

        rmseCV, maeCV, ccCV = [], [], []
        stepsRMSECV, stepsMAECV, stepsCCCV = [], [], []
        
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=config.SEED_VALUE)
        
        if num_selected > 0:  
        
            Tx_selected = Tx.loc[:, select].copy()
            
            for train_index, test_index in rkf.split(Tx_selected):
                X_train, X_test = Tx_selected.iloc[train_index], Tx_selected.iloc[test_index]
                y_train, y_test = Ty.iloc[train_index], Ty.iloc[test_index]

                X_train_timeseries = reshape_timeseries(X_train)
                X_test_timeseries = reshape_timeseries(X_test)

                random.seed(config.SEED_VALUE)
                np.random.seed(config.SEED_VALUE)
                tf.random.set_seed(config.SEED_VALUE)

                model = modelFunction(len(X_train.columns))  
                model.fit(X_train_timeseries, np.array(y_train).ravel(), verbose=0) 
                pred = model.predict(X_test_timeseries, verbose=0).ravel()
                rmse, mae, cc = calculate_errors(y_test, pred)
                
                rmseCV.append(rmse)
                maeCV.append(mae)
                ccCV.append(cc)
                
                dfSteps, _, _ = predictions_h_stepsahead_LSTM(X_test, y_test, model, config.N_STEPS)
                
                if dfSteps.empty:
                    continue

                stepsRMSECV.append(dfSteps['RMSE'])
                stepsMAECV.append(dfSteps['MAE'])
                stepsCCCV.append(dfSteps['CC'])
        
        entry = {
            'Run': n_seed,
            'Generations': n_generations,
            'RMSE MOEA': sol.objectives[0],
            'N': num_selected,
            'RMSE CV': np.round(rmseCV, 6),
            'MAE CV': np.round(maeCV, 6),
            'CC CV': np.round(ccCV, 6),
            'Mean RMSE CV': np.round(np.mean(rmseCV), 6),
            'Mean MAE CV': np.round(np.mean(maeCV), 6),
            'Mean CC CV': np.round(np.mean(ccCV), 6),
            'RMSE StepsAhead': np.round(stepsRMSECV, 6),
            'MAE StepsAhead': np.round(stepsMAECV, 6),
            'CC StepsAhead': np.round(stepsCCCV, 6),
            'Mean RMSE StepsAhead': np.round(np.mean(stepsRMSECV, axis=0), 6),
            'Mean MAE StepsAhead': np.round(np.mean(stepsMAECV, axis=0), 6),
            'Mean CC StepsAhead': np.round(np.mean(stepsCCCV, axis=0), 6),
            'SelectedAttrib': Tx.columns[select].to_numpy()
        }
        
        dfSolutions = pd.concat([dfSolutions, pd.DataFrame([entry])], ignore_index=True)

    return dfSolutions


def best_models_ML_test(Tx, Ty, Vx, Vy, df, modelFunction, colNames=None, is_classification=False):    
    """
    Train and test the best machine learning model on selected features,
    supporting both regression and classification.
    
    Parameters:
    Tx (DataFrame): Training feature matrix.
    Ty (Series): Training target variable.
    Vx (DataFrame): Test feature matrix.
    Vy (Series): Test target variable.
    df (DataFrame): Contains selected features and previous evaluation results.
    modelFunction (callable): ML model function.
    colNames (list): Names of the columns.
    is_classification (bool): Whether the problem is classification.
    
    Returns:
    DataFrame: Updated results with training and test metrics.
    """
    if colNames is None:
        colNames = ['Train RMSE', 'Train MAE', 'Train CC', 'Train RMSE StepsAhead', 'Train MAE StepsAhead', 'Train CC StepsAhead',
                    'Test RMSE', 'Test MAE', 'Test CC', 'Test RMSE StepsAhead', 'Test MAE StepsAhead', 'Test CC StepsAhead']
        
    metric1_train, metric2_train, metric3_train = [], [], []
    metric1_test, metric2_test, metric3_test = [], [], []
    stepsMetric1_train, stepsMetric2_train, stepsMetric3_train = [], [], []
    stepsMetric1_test, stepsMetric2_test, stepsMetric3_test = [], [], []


    if df['N'] > 0:
        select = df['SelectedAttrib']
        train_X_selectedA = Tx.loc[:, select]
        test_X_selectedA = Vx.loc[:, select]
            
        model = modelFunction.fit(train_X_selectedA.to_numpy(), Ty.to_numpy().ravel())
        pred_train = model.predict(train_X_selectedA.to_numpy())
        pred_test = model.predict(test_X_selectedA.to_numpy())

        metrics_train = calculate_errors(Ty.to_numpy(), pred_train, is_classification)
        metrics_test = calculate_errors(Vy.to_numpy(), pred_test, is_classification)
        
        metric1_train.append(metrics_train[0])
        metric2_train.append(metrics_train[1])
        metric3_train.append(metrics_train[2] if not is_classification else np.nan)

        metric1_test.append(metrics_test[0])
        metric2_test.append(metrics_test[1])
        metric3_test.append(metrics_test[2] if not is_classification else np.nan)

        dfSteps_train, _, _ = predictions_h_stepsahead(train_X_selectedA, Ty, model, config.N_STEPS, is_classification)
        dfSteps_test, _, _ = predictions_h_stepsahead(test_X_selectedA, Vy, model, config.N_STEPS, is_classification)

        stepsMetric1_train.append(dfSteps_train.iloc[:, 0])
        stepsMetric2_train.append(dfSteps_train.iloc[:, 1])
        stepsMetric3_train.append(dfSteps_train.iloc[:, 2] if not is_classification else np.nan)

        stepsMetric1_test.append(dfSteps_test.iloc[:, 0])
        stepsMetric2_test.append(dfSteps_test.iloc[:, 1])
        stepsMetric3_test.append(dfSteps_test.iloc[:, 2] if not is_classification else np.nan)


    results = pd.Series({
        colNames[0]: np.round(metric1_train, 6),
        colNames[1]: np.round(metric2_train, 6),
        colNames[2]: np.round(metric3_train, 6) if not is_classification else np.nan,
        colNames[3]: np.round(np.array(stepsMetric1_train).ravel(), 6),
        colNames[4]: np.round(np.array(stepsMetric2_train).ravel(), 6),
        colNames[5]: np.round(np.array(stepsMetric3_train).ravel(), 6) if not is_classification else np.nan,

        colNames[6]: np.round(metric1_test, 6),
        colNames[7]: np.round(metric2_test, 6),
        colNames[8]: np.round(metric3_test, 6) if not is_classification else np.nan,
        colNames[9]: np.round(np.array(stepsMetric1_test).ravel(), 6),
        colNames[10]: np.round(np.array(stepsMetric2_test).ravel(), 6),
        colNames[11]: np.round(np.array(stepsMetric3_test).ravel(), 6) if not is_classification else np.nan
    })

    return pd.concat([df, results])



def best_models_LSTM_test(Tx, Ty, Vx, Vy, df, modelFunction):    
    """
    Train and test the best LSTM model on selected features.
    
    Parameters:
    Tx (DataFrame): Training feature matrix.
    Ty (Series): Training target variable.
    Vx (DataFrame): Test feature matrix.
    Vy (Series): Test target variable.
    df (DataFrame): Contains selected features and previous evaluation results.
    modelFunction (callable): LSTM model function.
    
    Returns:
    DataFrame: Updated results with training and test metrics.
    """
    if df['N'] > 0:
        select = df['SelectedAttrib']
        train_X_selectedA = Tx.loc[:, select]
        test_X_selectedA = Vx.loc[:, select]

        X_train_timeseries = reshape_timeseries(train_X_selectedA)
        X_test_timeseries = reshape_timeseries(test_X_selectedA)


        random.seed(config.SEED_VALUE)
        np.random.seed(config.SEED_VALUE)
        tf.random.set_seed(config.SEED_VALUE)

        model = modelFunction(len(select))  
        model.fit(X_train_timeseries, np.array(Ty).ravel(), verbose=0) 
        pred_train = model.predict(X_train_timeseries, verbose=0).ravel()
        pred_test = model.predict(X_test_timeseries, verbose=0).ravel()

        rmse_train, mae_train, cc_train = calculate_errors(Ty.to_numpy(), pred_train)
        rmse_test, mae_test, cc_test = calculate_errors(Vy.to_numpy(), pred_test)

        df_steps_train, _, _ = predictions_h_stepsahead_LSTM(train_X_selectedA, Ty, model, config.N_STEPS)
        df_steps_test, _, _ = predictions_h_stepsahead_LSTM(test_X_selectedA, Vy, model, config.N_STEPS)

    results = pd.Series({
        'Train RMSE': round(rmse_train, 6),
        'Train MAE': round(mae_train, 6),
        'Train CC': round(cc_train, 6),
        'Train RMSE StepsAhead': np.round(df_steps_train['RMSE'].to_numpy(), 6),
        'Train MAE StepsAhead': np.round(df_steps_train['MAE'].to_numpy(), 6),
        'Train CC StepsAhead': np.round(df_steps_train['CC'].to_numpy(), 6),

        'Test RMSE': round(rmse_test, 6),
        'Test MAE': round(mae_test, 6),
        'Test CC': round(cc_test, 6),
        'Test RMSE StepsAhead': np.round(df_steps_test['RMSE'].to_numpy(), 6),
        'Test MAE StepsAhead': np.round(df_steps_test['MAE'].to_numpy(), 6),
        'Test CC StepsAhead': np.round(df_steps_test['CC'].to_numpy(), 6)
    })

    return pd.concat([df, results])
