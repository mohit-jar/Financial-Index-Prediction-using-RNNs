# Stock Price Prediction with RNNs

This project utilizes Recurrent Neural Networks (RNNs) to forecast stock index prices. It provides a complete pipeline from data preparation and model training to evaluation and visualization, comparing the performance of SimpleRNN, GRU, and LSTM architectures.

---
## Overview

The core of the project is to predict future values of the SH300IF stock index based on its historical price data. It implements three different stateful RNN models, trains them on a time-series dataset, and evaluates their predictive accuracy using metrics like Mean Squared Error (MSE) and Mean Directional Accuracy (MDA). The final output includes visualizations comparing the predicted prices against the actual prices.

---
## Features

* **Model Comparison**: Implements and compares three popular RNN architectures: SimpleRNN, GRU (Gated Recurrent Unit), and LSTM (Long Short-Term Memory).
* **Data Preprocessing**: A dedicated data preparation pipeline handles loading, standardization (Standard or Min-Max scaling), and transformation of time-series data into a supervised learning format.
* **Custom Metrics**: In addition to standard loss functions like MSE, the project uses custom metrics relevant to financial forecasting:
    * **RMSE (Root Mean Squared Error)**: Measures the magnitude of prediction errors.
    * **MDA (Mean Directional Accuracy)**: Measures the correctness of the predicted direction (up or down) of the stock price movement.
* **Stateful Models**: The RNNs are implemented as stateful models, which allows them to maintain state across batches when processing sequences. Data is trimmed to be divisible by the batch size to support this.
* **Callbacks**: Utilizes Keras callbacks for efficient training, including `ModelCheckpoint` to save the best model and `ReduceLROnPlateau` to adjust the learning rate dynamically.
* **Comprehensive Evaluation**: A `ModelPredictions` class handles prediction, de-standardization of results, metric calculation, and visualization of the model's performance on training, validation, and test datasets.
* **Visualization**: Generates clear plots of actual vs. predicted values, with labels that include performance scores for easy comparison.

---
## Code Structure

The project is organized into modular Python files:

* **`predict-stock-rnn.ipynb`**: The main Jupyter Notebook that orchestrates the entire workflow. It sets hyperparameters, loads data, builds and trains the models, and visualizes the final predictions.
* **`data-preparation.py`**: Contains the `StockIndexDataset` class responsible for loading raw data from files, performing standardization, splitting data into training, validation, and test sets, and structuring it for the RNNs.
* **`rnn-models.py`**: Defines the architectures for the SimpleRNN, GRU, and LSTM models. It also contains the `ModelPredictions` class for evaluation and the definitions for custom metrics (`rmse`, `mda`).
* **`utils.py`**: A utility script with helper functions for plotting the time-series data using `matplotlib` and formatting plot labels.

---
## Workflow

1.  **Data Loading**: The `StockIndexDataset` class reads historical stock data (date, time, open, close) from text files.
2.  **Preprocessing**: The raw closing price sequence is standardized. It is then transformed into input/output pairs, where a sequence of past values (`time_steps`) is used to predict a future value (`forecast_steps`).
3.  **Data Splitting**: The dataset is split into training, validation, and test sets without shuffling to preserve the temporal order.
4.  **Model Building**: Three Keras models (SimpleRNN, GRU, LSTM) are constructed, each with a recurrent layer, a dropout layer for regularization, and dense layers for the final output.
5.  **Training**: The models are trained on the training data, and the best-performing weights are saved based on validation loss.
6.  **Prediction & Evaluation**: After training, the `ModelPredictions` class is used to generate predictions on all data splits. The predictions are converted back to their original scale, and performance metrics (MSE, MDA) are calculated.
7.  **Visualization**: The results are plotted, showing the actual stock prices against the predicted values from each model on the different data splits. A final comparative plot shows the performance of all three models on the test set.

---
## Dependencies

To run this project, you will need the following libraries:

* pandas
* numpy
* matplotlib
* scikit-learn
* Keras / TensorFlow
