from flask import Flask, render_template, request
import joblib
import re
import nltk
import os
import gdown

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

def download_models():
    files = {
        'rf_baseline.pkl': '18UKcB-AAI1xIvZkPx7vzVyhCfJAUMjwm',
        'rf_model.pkl': '1IyGtU-y9wzGGk3WBryB8EcovHO6MGP9q',
        'selected_features.pkl': '11YejxSlrYDoxfOPSs22mWTihY9uXm6Mf',
        'tfidf_vectorizer.pkl': '1HjAePoTEVtQQ360m0ghh5GDFMmfcpXk5'
    }
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)
            print(f"{filename} downloaded!")

download_models()

rf_pso = joblib.load('rf_model.pkl')
rf_baseline = joblib.load('rf_baseline.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
selected_features = joblib.load('selected_features.pkl')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/baseline', methods=['GET', 'POST'])
def baseline():
    prediction = None
    confidence = None
    article = ''
    if request.method == 'POST':
        article = request.form['article']
        if article.strip() != '':
            cleaned = preprocess(article)
            vectorized = tfidf.transform([cleaned]).toarray()
            result = rf_baseline.predict(vectorized)[0]
            probability = rf_baseline.predict_proba(vectorized)[0]
            prediction = 'FAKE' if result == 1 else 'REAL'
            confidence = round(max(probability) * 100, 2)
    return render_template('baseline.html', prediction=prediction, confidence=confidence, article=article)

@app.route('/pso', methods=['GET', 'POST'])
def pso():
    prediction = None
    confidence = None
    article = ''
    if request.method == 'POST':
        article = request.form['article']
        if article.strip() != '':
            cleaned = preprocess(article)
            vectorized = tfidf.transform([cleaned]).toarray()
            vectorized_pso = vectorized[:, selected_features]
            result = rf_pso.predict(vectorized_pso)[0]
            probability = rf_pso.predict_proba(vectorized_pso)[0]
            prediction = 'FAKE' if result == 1 else 'REAL'
            confidence = round(max(probability) * 100, 2)
    return render_template('pso.html', prediction=prediction, confidence=confidence, article=article)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
