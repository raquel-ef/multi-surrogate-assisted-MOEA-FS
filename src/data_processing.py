from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import config.config as config


def read_arff(path):
    """
    Reads an ARFF file and converts it into a pandas DataFrame.

    Parameters:
    path (str): File path to the ARFF dataset.

    Returns:
    DataFrame: Loaded dataset.
    """
    data = arff.loadarff(path)
    return pd.DataFrame(data[0])


def split_features_target(df):
    """
    Splits a DataFrame into features (X) and target (Y).

    Parameters:
    df (DataFrame): Dataset to be split.

    Returns:
    tuple: Feature set (X) and target set (Y).
    """
    return df.iloc[:, :-1], df.iloc[:, -1:]


def reshape_timeseries(data):
    """
    Reshapes data for time series input (LSTM compatibility).

    Parameters:
    data (np.ndarray): Input data.

    Returns:
    np.ndarray: Reshaped data.
    """
    return np.reshape(data, (data.shape[0], 1, data.shape[1]))


def to_numpy(df):
    """
    Converts a pandas DataFrame to a NumPy array.

    Parameters:
    df (DataFrame): Input DataFrame.

    Returns:
    np.ndarray: Converted NumPy array.
    """
    return np.asarray(df)


def preprocess_data(df):
    """
    Preprocesses data by splitting into training and test sets,
    normalizing the data, and reshaping it for LSTM.

    Parameters:
    df (DataFrame): Input dataset.

    Returns:
    dict: Contains processed time series and normalized data splits.
    """
    # Split data into training (80%) and test (20%) sets
    n = len(df)
    train_set = df.iloc[: int(0.8 * n)]
    test_set = df.iloc[int(0.8 * n) :]

    # Extract features and targets
    train_X, train_Y = split_features_target(train_set)
    test_X, test_Y = split_features_target(test_set)

    # Normalize data using MinMaxScaler
    scaler_X, scaler_Y = preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()
    train_X = scaler_X.fit_transform(train_X)
    train_Y = scaler_Y.fit_transform(train_Y)
    test_X = scaler_X.transform(test_X)
    test_Y = scaler_Y.transform(test_Y)

    # Convert to numpy arrays
    train_X, train_Y = to_numpy(train_X), to_numpy(train_Y)
    test_X, test_Y = to_numpy(test_X), to_numpy(test_Y)

    # Reshape data for LSTM
    train_X_timeseries = reshape_timeseries(train_X)
    test_X_timeseries = reshape_timeseries(test_X)

    # Convert normalized data back to DataFrame
    train_X_df = pd.DataFrame(train_X, columns=train_set.columns[:-1])
    train_Y_df = pd.DataFrame(train_Y, columns=[train_set.columns[-1]])
    test_X_df = pd.DataFrame(test_X, columns=test_set.columns[:-1])
    test_Y_df = pd.DataFrame(test_Y, columns=[test_set.columns[-1]])

    return {
        "timeseries": (train_X_timeseries, train_Y, test_X_timeseries, test_Y),
        "normalized": (train_X_df, train_Y_df, test_X_df, test_Y_df)
    }


def preprocess_data_classification(df):
    """
    Preprocesses classification data by splitting into training and test sets,
    normalizing the data, and reshaping it for LSTM.

    Parameters:
    df (DataFrame): Input dataset.

    Returns:
    dict: Contains processed time series, normalized data splits and
    label encoder data.
    """
    # Split data into training (80%) and test (20%) sets
    X, y = split_features_target(df)

    np.random.seed(config.SEED_VALUE)
    train_X, test_X, train_Y, test_Y = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.2)

    # Encode categorical target variable
    label_encoder = LabelEncoder()
    train_Y = label_encoder.fit_transform(train_Y)
    test_Y = label_encoder.transform(test_Y)

    # Encode categorical features if any
    categorical_cols = train_X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = preprocessing.OrdinalEncoder()
        train_X[categorical_cols] = encoder.fit_transform(train_X[categorical_cols])
        test_X[categorical_cols] = encoder.transform(test_X[categorical_cols])

    # Normalize feature data using MinMaxScaler
    scaler_X = preprocessing.MinMaxScaler()
    train_X = scaler_X.fit_transform(train_X)
    test_X = scaler_X.transform(test_X)

    # Convert normalized data back to DataFrame
    train_X_df = pd.DataFrame(train_X, columns=X.columns)
    train_Y_df = pd.DataFrame(train_Y, columns=y.columns)
    test_X_df = pd.DataFrame(test_X, columns=X.columns)
    test_Y_df = pd.DataFrame(test_Y, columns=y.columns)

    # Reshape data for LSTM (samples, timesteps, features)
    train_X_timeseries = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X_timeseries = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    return {
            "timeseries": (train_X_timeseries, train_Y, test_X_timeseries, test_Y),
            "normalized": (train_X_df, train_Y_df, test_X_df, test_Y_df),
            "label_encoder": label_encoder
        }