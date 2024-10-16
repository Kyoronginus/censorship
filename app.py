import nltk
import torch
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load pre-trained censorship model
classifier = pipeline('text-classification', model='unitary/toxic-bert')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_text():
    input_text = request.form['text']
    result = classifier(input_text)[0]  # Get model result
    return jsonify({
        'text': input_text,
        'label': result['label'],
        'score': result['score']
    })

if __name__ == '__main__':
    app.run(debug=True)
