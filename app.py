import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import os
import numpy as np
import random
import json
from keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for, jsonify, g, session
import mysql.connector
from mysql.connector import Error
from flask_bcrypt import Bcrypt
from flask_cors import CORS

# Load the trained model and supporting files
model = load_model('keras_intent_model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    # Tokenize the sentence and lemmatize each word
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Convert sentence to bag of words
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    # Predict the class of the sentence
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    # Get the response for the predicted intent
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if len(ints) == 0:  # Check if no intents are predicted
        return "Sorry, I didn't quite understand that. Could you try again?"
    res = get_response(ints, intents)  # Ensure correct function name is used
    return res

# Flask application setup
app = Flask(__name__)
app.static_folder = 'static'
CORS(app)  # Allow frontend requests
bcrypt = Bcrypt(app)
app.secret_key = os.urandom(24)

### Establishing database connection setup
def get_db_connection():
    if 'conn' not in g:
        try:
            g.conn = mysql.connector.connect(
                host="127.0.0.1",
                user="root",
                password="Pandu@7463",
                database="chatbot",
                auth_plugin='mysql_native_password'
            )
            g.cursor = g.conn.cursor(dictionary=True)
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None, None
    return g.conn, g.cursor

@app.teardown_appcontext
def close_db_connection(exception):
    conn = g.pop('conn', None)
    cursor = g.pop('cursor', None)
    if cursor:
        cursor.close()
    if conn:
        conn.close()

## User login routing
@app.route('/login', methods=['GET', 'POST'])
def user_login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn, cursor = get_db_connection()
        if not conn or not cursor:
            return "Database connection error", 500

        cursor.execute(
            "SELECT * FROM users WHERE username = %s AND password = %s",
            (username, password)
        )
        user = cursor.fetchone()

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('user_home'))
        else:
            error = "Invalid username or password"
    return render_template('user_login.html', error=error)
##
## User registration routing
@app.route('/register', methods=['GET', 'POST'])
def user_register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email') 
        password = request.form.get('password') 
        
        conn, cursor = get_db_connection()
        if not conn or not cursor:
            return "Database connection error", 500

        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, password)
            )
            conn.commit()
            return redirect(url_for('user_login'))
        except mysql.connector.IntegrityError:
            return render_template('user_registration.html', error="Username or email already exists")
    return render_template('user_registration.html')

# Home Route (Chatbot interface)
@app.route("/")
def home():
    return render_template("index.html")

# Chatbot Response Route
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    return render_template("user_login.html")

@app.route("/user_home")
def user_home():
    return render_template("user_home.html")

@app.route("/book-appointment")
def book_appointment():
    return render_template("appointment.html")

# Logout Route (Optional)
@app.route("/logout")
def logout():
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
