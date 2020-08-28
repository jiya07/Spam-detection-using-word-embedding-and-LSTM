import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, GlobalMaxPool1D
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('spam_or_not_spam.csv')
print(df.head())
df = df.dropna()

X = df['email']
y = df['label'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

MaxVocabSize = 2000
tokenizer = Tokenizer(num_words=MaxVocabSize)
tokenizer.fit_on_texts(x_train)
sequencesTrain = tokenizer.texts_to_sequences(x_train)
sequencesTest = tokenizer.texts_to_sequences(x_test)

V = len(tokenizer.word_index)
print(V)

maxLen = 50
X_train = pad_sequences(sequencesTrain, maxlen = maxLen)
print(X_train.shape)
X_test = pad_sequences(sequencesTest, maxlen=maxLen)

model = Sequential([
    Embedding(V+1,20,input_length=maxLen),
    LSTM(15, return_sequences = True),
    GlobalMaxPool1D(),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
r = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_data=(X_test, y_test))

plt.plot(r.history['loss'], label = 'Loss')
plt.plot(r.history['val_loss'], label = 'Val_loss')
plt.legend()
print(plt.show())

plt.plot(r.history['accuracy'], label = 'Accuracy')
plt.plot(r.history['val_accuracy'], label = 'Val_accuracy')
plt.legend()
print(plt.show())

model.save('spam_detection.h5')
pickle.dump(tokenizer, open("tokenizer.pickle", "wb"))