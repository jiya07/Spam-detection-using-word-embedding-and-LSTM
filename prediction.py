from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

model = load_model('spam_detection.h5')
loaded_tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

txt="Congratulations! You have a been a lucky customer"
maxLen = 50
seq= loaded_tokenizer.texts_to_sequences([txt])
padded = pad_sequences(seq, maxlen = maxLen)
pred = model.predict_classes(padded)
if pred > 0:
    t="It is a spam message"
else:
    t="It is not spam"
print(t)