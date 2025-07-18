import nltk
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nlp_processing import words, classes, documents

# Load the data (processed from nlp_processing.py)
X_train = []
y_train = []

for doc in documents:
    # Tokenize each word
    word_bag = [1 if w in nltk.word_tokenize(doc[0].lower()) else 0 for w in words]
    X_train.append(word_bag)
    y_train.append(classes.index(doc[1]))  # Label encoding of the tag

# Convert lists to arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(X_train[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

# Compile and train the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=8)

# Save the model
model.save("E:/Placements/Projects/GIT_UPLOADS/chatbot-nlp/models/chatbot_model.h5")
