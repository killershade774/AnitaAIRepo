# src/predict.py

from anita import AnitaAI
import numpy as np

# Initialize Anita and load the model
anita = AnitaAI()
anita.load_model('../models/anita_model.h5')

# Example prediction
X_test = np.random.rand(1, 10, 1)
prediction = anita.predict(X_test)
print("Prediction:", prediction)