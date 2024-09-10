# Stock Market Price Prediction

## Introduction
This project aims to predict the Adjusted Close prices of AstraZeneca's stock using a multi-layer Long Short-Term Memory (LSTM) Recurrent Neural Network model. The model is trained on five years of historical stock data, leveraging the ability of LSTM networks to store information over time, which is particularly useful for time series data.

## Columns in the Dataset
- **Open**: The price at which the stock started trading at the beginning of the trading day.
- **High**: The highest price at which the stock traded during the trading day.
- **Low**: The lowest price at which the stock traded during the trading day.
- **Close**: The price at which the stock ended trading at the end of the trading day.
- **Adj Close**: The closing price adjusted for corporate actions such as dividends, stock splits, and new stock offerings.
- **Volume**: The number of shares that were traded during the trading day.

## What I Did
### Data Collection
Collected five years of historical stock data for AstraZeneca from Yahoo! Finance using the yfinance API.

### Data Preprocessing
- Converted the data into a pandas DataFrame.
- Normalized the data using MinMaxScaler to scale the values between 0 and 1, which helps improve the performance of the model.

### Training Data Preparation
- Created a training dataset by splitting the data into training and testing sets (80% training, 20% testing).
- Created sequences of 60 time-steps for the LSTM model.

### Model Building
- Built a Sequential model with four LSTM layers, each with 50 neurons and tanh activation function.
- Added Dropout layers after each LSTM layer to prevent overfitting.
- Compiled the model using the Mean Squared Error loss function and the Adam optimizer.

### Model Training
- Trained the model on the training data for 200 epochs with a batch size of 64.

### Model Evaluation
- Evaluated the model's performance using the Root Mean Squared Error (RMSE) metric.
- Plotted the predicted vs. actual Adjusted Close prices to visualize the model's performance.

## Results
The model provided good results with a relatively low RMSE, indicating that it was able to predict the Adjusted Close prices with reasonable accuracy. The predicted prices were close to the actual prices, as visualized in the plot of predicted vs. actual values. Further improvements could be made by experimenting with different numbers of layers, epochs, and batch sizes.
