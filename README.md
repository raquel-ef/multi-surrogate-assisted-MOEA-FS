# Multi-surrogate assisted multi-objective evolutionary algorithms for feature selection in regression and classification problems with time series data

## Overview
Feature selection (FS) wrapper methods are powerful mechanisms for reducing the complexity of prediction models while preserving or even improving their precision. However, traditional multi-objective evolutionary algorithms (MOEAs) for wrapper-based FS can be computationally expensive, particularly in high-dimensional problems where deep learning models are used.

To address this challenge, this project introduces: 
- Multi-surrogate assisted MOEA for FS, which enhances generalization error while maintaining computational efficiency. 
- Support for multiple machine and deep learning models, including:

    - Random Forest (RF)
    - Support Vector Machines (SVM)
    - Long Short-Term Memory (LSTM)

The approach has been evaluated on regression and classification problems using time series data for air quality forecasting and indoor temperature prediction.


## Project Structure
``` plaintext
📂 config/                      # Configuration file
📂 data/                        # Dataset storage
📂 models/                      # Trained models
📂 notebooks/                   # Jupyter Notebooks with examples
 ├── Multi-surrogate Classification.ipynb
 ├── Multi-surrogate Regression.ipynb
 ├── Multi-surrogate training classification.ipynb
 ├── Multi-surrogate training regression.ipynb
📂 problems/                    # Problem-specific implementations
 ├── Multi_Surrogate_FS_LSTM_Classification.py
 ├── Multi_Surrogate_FS_LSTM.py
 ├── Multi_Surrogate_FS_ML_Classification.py
 ├── Multi_Surrogate_FS_ML.py
 ├── Wrapper_LSTM.py
 ├── Wrapper_ML_Classification.py
 ├── Wrapper_ML.py
📂 src/                         # Source code
 ├── data_processing.py             # Preprocessing and cleaning
 ├── evaluation.py                  # Model evaluation scripts
 ├── utils.py                       # Utility functions
📂 variables/               # Variable storage
📜 requirements.txt         # Dependencies
```

## Installation
Ensure you have Python 3.10 installed and install the following dependencies:
```sh
pip install -r requirements.txt
```


## Usage
1. **Prepare the dataset**. Place data in `.arff` format in the [data](/data/) directory. Note that the data must be previously transformed using a sliding window method (see function `lags` in [utils](/src/utils.py) for this transformation).
2. **Train multiple surrogate models**. Train the surrogate models as in the example notebook [Multi-surrogate training regression](/notebooks/Multi-surrogate%20training%20regression.ipynb) for regression or [Multi-surrogate training classification](/notebooks/Multi-surrogate%20training%20classification.ipynb) for classification. The parameters should be previously configured in [config](/config/config.py).
3. **Comparisons**. Run and compare multi-surrogate assisted MOEA and wrapper methods for FS as in the example notebooks ([Multi-surrogate Regression](/notebooks/Multi-surrogate%20Regression.ipynb) or  [Multi-surrogate Classification](/notebooks/Multi-surrogate%20Classification.ipynb)).



## Citation
If you use this software in your work, please include the following citation:
```
@article{espinosa2023multi,
  title={Multi-surrogate assisted multi-objective evolutionary algorithms for feature selection in regression and classification problems with time series data},
  author={Espinosa, Raquel and Jim{\'e}nez, Fernando and Palma, Jos{\'e}},
  journal={Information Sciences},
  volume={622},
  pages={1064--1091},
  year={2023},
  publisher={Elsevier}
}
```

## License
[MIT License](/LICENSE)


