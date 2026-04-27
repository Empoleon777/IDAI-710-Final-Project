import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

class MyLSTM():
    def __init__(self):
        model = Sequential(
            LSTM(128),
            Dropout(0.2),
            LSTM(128),
            Dropout(0.2),
            Dense(1)
        )

        model.compile(
            optimizer='adam', 
            loss='mean_squared_error'
        )

    def Train(self, X_train, y_train, epochs, batch_size, validation_split):
        self.X_train = X_train
        self.y_train = y_train
        self.training_data = self.model.fit(X_train, y_train, epochs, batch_size, validation_split)
    
    def Test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = self.model.predict(X_test)
        self.rmse = np.sqrt(np.mean((y_test - self.predictions)**2))
        self.mae = mean_absolute_error(y_test, self.predictions)
        # print(f"Accuracy: {accuracy_score(y_test, predictions)}")
        # print(f"Other Metrics: {classification_report(y_test, predictions)}")
    
    def VisualizeActualVsPredicted(self, title):
        x = self.X_test.columns.tolist()

        plt.figure(figsize=(12, 6))
        plt.plot(x[0], self.y_test, label='Actual Demand')
        plt.plot(x[0], self.predictions, label='Predicted Demand')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Demand (Passengers)')
        plt.legend()
        plt.savefig('../Graphs/actualvspred.jpeg')
        plt.show()

    def VisualizeLoss(self, title):
        plt.figure(figsize=(12, 6))
        plt.plot(self.training_data.history['loss'], label='Training Loss')
        plt.plot(self.training_data.history['val_loss'], label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.savefig('../Graphs/loss.jpeg')
        plt.show()

    def WritePredsToCSV(self):
        output = self.X_test[['datetime', 'demand']].copy()
        output.rename(columns={'demand': 'actual_demand'}, inplace=True)
        output['predicted_demand'] = self.predictions
        output.to_csv('../Data/actual_vs_predicted.csv', index=False)

data = pd.read_csv('Data/data_with_weather.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
demand = data['demand'].astype(float).values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(demand)

range_2020 = 7270
range_2021 = 16030
range_2022 = 24790
range_2023 = 28417


X_2020 = scaled_data.loc(scaled_data[0:range_2020, ['datetime', 'hour', 'day_of_week', 'is_weekend', 'event', 'temp', 'feelslike', 'precip', 'snow', 'snowdepth', 'is_snowing', 'is_raining']])
y_2020 = scaled_data.loc(scaled_data[0:range_2020, ['demand']])

X_2021 = scaled_data.loc(scaled_data[range_2020+1:range_2021, ['datetime', 'hour', 'day_of_week', 'is_weekend', 'event', 'temp', 'feelslike', 'precip', 'snow', 'snowdepth', 'is_snowing', 'is_raining']])
y_2021 = scaled_data.loc(scaled_data[range_2020+1:range_2021, ['demand']])

X_2022 = scaled_data.loc(scaled_data[range_2021+1:range_2022, ['datetime', 'hour', 'day_of_week', 'is_weekend', 'event', 'temp', 'feelslike', 'precip', 'snow', 'snowdepth', 'is_snowing', 'is_raining']])
y_2022 = scaled_data.loc(scaled_data[range_2021+1:range_2022, ['demand']])

X_2023 = scaled_data.loc(scaled_data[range_2022+1:range_2023, ['datetime', 'hour', 'day_of_week', 'is_weekend', 'event', 'temp', 'feelslike', 'precip', 'snow', 'snowdepth', 'is_snowing', 'is_raining']])
y_2023 = scaled_data.loc(scaled_data[range_2022+1:range_2023, ['demand']])

X_2020_train, X_2020_test, y_2020_train, y_2020_test = train_test_split(
    X_2020, y_2020, test_size=0.2, shuffle=False
)

X_2021_train, X_2021_test, y_2021_train, y_2021_test = train_test_split(
    X_2021, y_2021, test_size=0.2, shuffle=False
)

X_2022_train, X_2022_test, y_2022_train, y_2022_test = train_test_split(
    X_2022, y_2022, test_size=0.2, shuffle=False
)

X_2023_train, X_2023_test, y_2023_train, y_2023_test = train_test_split(
    X_2023, y_2023, test_size=0.2, shuffle=False
)