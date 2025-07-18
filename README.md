# Chatbot with NLP

This is a simple chatbot built using NLP techniques like tokenization, lemmatization, and machine learning for intent classification. The chatbot is implemented using Flask and TensorFlow.

## Setup

1. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the training scripts and the app.py

3. **How to Run**

    1. **Set Up Environment**:
   -     Install dependencies:
         ```bash
         pip install -r requirements.txt
         ```
    
2. **Train the Model**:
   - Run the training script to train the model with your `intents.json` data:
     ```bash
     python src/training/train_model.py
     ```

3. **Run the Flask App**:
   - After training, run the Flask app:
     ```bash
     python app.py
     ```

4. **Interact with the Chatbot**:
   - Send POST requests to `http://localhost:5000/chat` to interact with your chatbot.
   - Use a tool like **Postman** or **cURL** for testing:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"message": "Hi"}' http://localhost:5000/chat
     ```

Now you’re all set, bruh! Let me know if you need any adjustments or run into issues!
