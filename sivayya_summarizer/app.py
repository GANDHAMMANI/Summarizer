from flask import Flask, render_template, request, jsonify
import re
import spacy
import nltk
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')

def preprocess(sentences):
    preprocessed_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        extracted_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        preprocessed_sentences.append(" ".join(extracted_words))
    return preprocessed_sentences

def summarize_text(text, num_sentences):
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'"', '', text)
    original_sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = preprocess(original_sentences)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(preprocessed_sentences)
    sum_scores = matrix.toarray().sum(axis=1)
    ranked_scores = (-sum_scores).argsort()
    top_score_indices = sorted(ranked_scores[:num_sentences])
    final_sentences = [original_sentences[i] for i in top_score_indices]
    summary = " ".join(final_sentences)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text_input = data['text']
    num_sentences = int(data['num_sentences'])
    summary = summarize_text(text_input, num_sentences)
    formatted_summary = "\n" + "\n".join(summary.split(". "))
    return jsonify({'summary': formatted_summary})

if __name__ == '__main__':
    app.run(debug=True)
