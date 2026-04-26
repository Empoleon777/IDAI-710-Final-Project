import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy

from keras.models import Sequential
from keras.layers import LSTM, Dense

class LSTM():
    def __init__(self):
        look_back = 1
        model = Sequential(
            LSTM(16, input_shape=(1, look_back)),
            LSTM(8, input_shape=(1, look_back)),
            LSTM(4, input_shape=(1, look_back)),
            LSTM(2, input_shape=(1, look_back)),
            Dense(1)
        )

        model.compile(optimizer='adam', loss='mean_squared_error')

    def forward(self, X):
        logits = self.model(X)
        return logits
    
data = pd.read_csv('Data/data_with_weather.csv')
