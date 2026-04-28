import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os


class MyLSTM():
    def __init__(self):
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(12, 8), recurrent_dropout=0.1),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])

        self.model.compile(
            optimizer='adam', 
            loss='mean_squared_error'
        )

    def Train(self, X_train, y_train, dates_train, epochs, batch_size, validation_split):
        self.X_train = X_train
        self.y_train = y_train
        self.dates_train = dates_train
        self.training_data = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    
    def Test(self, X_test, y_test, dates_test):
        self.X_test = X_test
        self.y_test = y_test
        self.dates_test = dates_test
        self.predictions = self.model.predict(X_test)
        self.rmse = np.sqrt(np.mean((y_test - self.predictions)**2))
        self.mae = mean_absolute_error(y_test, self.predictions)
    
    def VisualizeActualVsPredicted(self, title):
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates_test, self.y_test, label='Actual Demand')
        plt.plot(self.dates_test, self.predictions, label='Predicted Demand')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Demand (Passengers)')
        plt.legend()
        os.makedirs('../Graphs', exist_ok=True)
        plt.savefig('Graphs/actualvspred.jpeg')
        plt.show()

    def VisualizeLoss(self, title):
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_data.history['loss'], label='Training Loss')
        plt.plot(self.training_data.history['val_loss'], label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        os.makedirs('../Graphs', exist_ok=True)
        plt.savefig('Graphs/loss.jpeg')
        plt.show()

    def WritePredsToCSV(self):
        output = pd.DataFrame({
            'datetime': self.dates_test,
            'actual_demand': self.y_test.flatten(),
            'predicted_demand': self.predictions.flatten()
        })
        os.makedirs('../Results', exist_ok=True)
        output.to_csv('Results/actual_vs_predicted_lstm.csv', index=False)
    
    def PrintMetrics(self):
        print(f"RMSE: {self.rmse}")
        print(f"MAE: {self.mae}")

def create_sequences(df, scaled, feature_cols, lookback=12):
    X = []
    y = []
    dates = []

    for i in range(lookback, len(df)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i][0])
        dates.append(df.index[i])

    return np.array(X), np.array(y), np.array(dates)

data = pd.read_csv('Data/data_with_weather.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
demand = data['demand'].astype(float).values.reshape(-1, 1)

features = [
    'demand',
    'temp',
    'feelslike',
    'precip',
    'snow',
    'snowdepth',
    'is_raining',
    'is_snowing'
]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[features])

X, y, target_dates = create_sequences(data, scaled_data, features, lookback=12)

X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, target_dates, test_size=0.2, shuffle=False
)

model = MyLSTM()
model.Train(X_train, y_train, dates_train, 250, 32, 0.1)
model.Test(X_test, y_test, dates_test)
model.VisualizeActualVsPredicted("Actual Demand vs. Predicted Demand (LSTM)")
model.VisualizeLoss("LSTM Training and Validation Loss")
model.WritePredsToCSV()
model.PrintMetrics()