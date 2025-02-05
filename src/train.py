import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess the dataset
def load_data(file_path, encoding='utf-8'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Permission denied to read the file {file_path}.")
    
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        # Try a different encoding if UTF-8 fails
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            lines = file.readlines()
    return lines

def preprocess_data(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    padded_sequences = pad_sequences(sequences, padding='post')
    return padded_sequences, tokenizer

def create_training_data(padded_sequences, seq_length):
    X = []
    y = []
    for seq in padded_sequences:
        if len(seq) > seq_length:  # Ensure sequence length is greater than seq_length
            for i in range(seq_length, len(seq)):
                X.append(seq[i-seq_length:i])
                y.append(seq[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Example usage
file_path = r'C:\Users\Mr.Popo\AnitaAi\data\cornell_movie_quotes_corpus\moviequotes.scripts.txt'  # Use raw string for file path
try:
    lines = load_data(file_path)
    print("File loaded successfully.")
    padded_sequences, tokenizer = preprocess_data(lines)
    seq_length = 10
    X_train, y_train = create_training_data(padded_sequences, seq_length)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Train the model
    anita_ai = AnitaAI()
    anita_ai.train(X_train, y_train, epochs=10, batch_size=32)

    # Generate responses
    def generate_response(model, tokenizer, input_text, seq_length):
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=seq_length, padding='post')
        input_seq = input_seq.reshape((1, seq_length, 1))
        predicted_seq = model.predict(input_seq)
        predicted_word = tokenizer.index_word[np.argmax(predicted_seq)]
        return predicted_word

    # Example usage
    input_text = "Hello, how are you?"
    response = generate_response(anita_ai.model, tokenizer, input_text, seq_length)
    print(response)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()