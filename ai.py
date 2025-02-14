from flask import Flask, render_template, request
import sqlite3
import datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__, template_folder='templates')

# Initialize AI model
symptoms_data = pd.DataFrame({
    'symptom': ['fever', 'headache', 'cough', 'fatigue', 'nausea'],
    'diagnosis': ['flu', 'migraine', 'cold', 'anemia', 'food poisoning']
})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms_data['symptom'])
model = MultinomialNB()
model.fit(X, symptoms_data['diagnosis'])

# Database Setup
def init_db():
    conn = sqlite3.connect('health_assistant.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS health_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        age INTEGER,
                        symptom TEXT,
                        diagnosis TEXT,
                        created_at TIMESTAMP
                    )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    name = request.form['name']
    age = request.form['age']
    symptom = request.form['symptom']

    # AI-powered diagnosis using ML model
    symptom_vec = vectorizer.transform([symptom.lower()])
    prediction = model.predict(symptom_vec)[0]
    diagnosis = f'Our AI diagnosis suggests: {prediction}. Please consult a doctor for confirmation.'
    
    conn = sqlite3.connect('health_assistant.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO health_records (name, age, symptom, diagnosis, created_at) VALUES (?, ?, ?, ?, ?)',
                   (name, age, symptom, diagnosis, datetime.datetime.now()))
    conn.commit()
    conn.close()

    return render_template('result.html', name=name, age=age, symptom=symptom, diagnosis=diagnosis)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
