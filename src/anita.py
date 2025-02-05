import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

class AnitaAI:
    
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            LSTM(128, input_shape=(None, 1), return_sequences=True),
            LSTM(128, return_sequences=False),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
    def load_model(self, model_path):
        self.model = tf.keras.models.load_model("C:\\Users\\Mr.Popo\\AnitaAi\\models")
    def predict(self, X):
        return self.model.predict(X)
