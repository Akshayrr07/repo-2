import json
import random
from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import nltk
import numpy as np
from nlp_processing import words, classes

# Load the trained model
model = tf.keras.models.load_model('models/chatbot_model.h5')

# Load the pickled data
intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))

# Initialize the app
app = Flask(__name__)

# Function to preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words] # type: ignore
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return(np.array(bag))

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json['message']
    # Predict the category
    p = bow(message, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_class = classes[results[0][0]]
    response = random.choice([i['responses'] for i in intents['intents'] if i['tag'] == return_class])
    
    return jsonify({"response": response[0]})

if __name__ == "__main__":
    app.run(debug=True)
