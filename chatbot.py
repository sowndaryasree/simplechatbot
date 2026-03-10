import json
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Load JSON data
with open("m.json") as file:
    data = json.load(file)

sentences = []
labels = []
responses = {}

# Extract data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Padding
max_len = max(len(x) for x in sequences)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

vocab_size = len(tokenizer.word_index) + 1

# Build Model
model = Sequential()
model.add(Embedding(vocab_size, 16, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(len(set(labels)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(padded, labels, epochs=500, verbose=0)

print("Chatbot Ready! Type 'quit' to exit.")

# Chat loop
while True:

    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')

    prediction = model.predict(padded_seq)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])

    response = random.choice(responses[tag[0]])
    print("Bot:", response)