# Alwan_Algifari_-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import ardi

# Load your dataset
# Misalnya, dataset Anda memiliki dua kolom: 'review' dan 'sentiment'
# Sesuaikan dengan struktur dataset yang Anda miliki
dataset = pd.read_csv('dataset_tempat_wisata_gorontalo.csv')

# Preprocessing
le = LabelEncoder()
dataset['sentiment_encoded'] = le.fit_transform(dataset['sentiment'])
X_train, X_test, y_train, y_test = train_test_split(dataset['review'], dataset['sentiment_encoded'], test_size=0.2, random_state=42)

# Tokenization
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Model
embedding_dim = 50
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(units=100))
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_pad, y_train, epochs=5, validation_data=(X_test_pad, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
