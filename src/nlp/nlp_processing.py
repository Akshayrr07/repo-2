import pickle
import nltk
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents
with open('data/intents.json') as file:
    intents = json.load(file)

# Initialize the array for words and class labels
words = []
classes = []
documents = []

# Tokenize the words and prepare the training data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((pattern, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lemmatize and sort words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words] # type: ignore
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save the data
pickle.dump(words, open("models/words.pkl", "wb"))
pickle.dump(classes, open("models/classes.pkl", "wb"))
