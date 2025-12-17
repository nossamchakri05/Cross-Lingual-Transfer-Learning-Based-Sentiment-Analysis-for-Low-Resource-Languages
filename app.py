from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from deep_translator import GoogleTranslator
import fasttext
import networkx as nx
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
import pymysql

# Initialize Flask App (Only once!)
app = Flask(__name__)
app.secret_key = "your_secret_key"  # ✅ Set secret key immediately

# File upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Database connection configuration
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'Tanmay@123'
DB_NAME = 'sentiment_db'



def get_db_connection():
    try:
        conn = pymysql.connect(
            host= DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except pymysql.MySQLError as err:
        print(f"DB connection error: {err}")
        return None

LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'hi': 'Hindi',
    # Add other languages as needed
}

# Load the FastText language detection model
fasttext_model = fasttext.load_model(r"E:\PROJECTS\nlp - Copy\lid.176.bin")

# Load the language similarity graph
G = nx.read_gml(r"E:\PROJECTS\nlp - Copy\language_similarity_graph.gml")

# ✅ Set device first
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the sentiment analysis model and tokenizer
sentiment_model_path = './sentiment_model'
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path)
tokenizer = BertTokenizer.from_pretrained(sentiment_model_path)

# Load French sentiment model and tokenizer
french_model_path = './french_model'
french_model = BertForSequenceClassification.from_pretrained(french_model_path)
french_tokenizer = BertTokenizer.from_pretrained(french_model_path)
french_model.to(device)
french_model.eval()

hindi_tokenizer = AutoTokenizer.from_pretrained("hindi_model")
hindi_model = AutoModelForSequenceClassification.from_pretrained("hindi_model").to(device)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentiment_model.to(device)
sentiment_model.eval()

# Preprocessing function for sentiment analysis
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

language_map = {
    'bn': 'bengali',
    'or': 'odia',
    'af': 'afrikaans',
    'ms': 'malay',
    'ur': 'urdu',
    'en': 'english',
    'hi': 'hindi',
    'fr': 'french',
    'es': 'spanish',
    'de': 'german',
    'ar': 'arabic'
}

high_resource = ['en', 'hi', 'fr', 'es', 'de', 'ar']

# Global variable to store original input
original_input_text = ""

def detect_language(text):
    prediction = fasttext_model.predict(text.strip())
    language = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]
    return language, confidence

def find_high_resource_match(language_code):
    if language_code in high_resource:
        return language_code

    if language_code not in language_map:
        print(f"Error: {language_code} not found in language map!")
        return None

    sims = []
    low_node = language_map[language_code]

    for high_lang in high_resource:
        high_node = language_map[high_lang]
        if G.has_node(low_node) and G.has_node(high_node):
            try:
                sim = G[low_node][high_node]['weight']
                sims.append((high_lang, sim))
            except KeyError:
                continue

    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims[0][0] if sims else None
def log_sentiment_result(user_id, input_sentence, translated_sentence, sentiment, sentiment_score):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                query = """
                    INSERT INTO sentiment_logs (user_id, input_sentence, translated_sentence, sentiment, sentiment_score, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute(query, (user_id, input_sentence, translated_sentence, sentiment, sentiment_score, timestamp))
                conn.commit()
    except Exception as e:
        print(f"Error logging sentiment result: {e}")
    finally:
        if conn:
            conn.close()

# Routes
@app.route('/')
def home():
    return render_template('reg.html')

# Registration Route
@app.route('/registration', methods=['GET', 'POST'])
def registration_page():
    if request.method == 'POST':
        name = request.form['name1']
        email = request.form['email']
        phone_number = request.form['contactNumber']
        password = request.form['password1']
        confirm_password = request.form['confirmPassword']

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for('registration_page'))

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                query = """
                INSERT INTO users (name, email, phone_number, password)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(query, (name, email, phone_number, hashed_password))
                conn.commit()
                cursor.close()
                conn.close()
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for('home'))
            else:
                flash("Database connection failed.", "danger")
        except mysql.connector.Error as err:
            flash(f"Database error: {err}", "danger")

    return render_template('registration.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('name')
    password = request.form.get('password')
    print(username)
    print(password)

    # Admin login logic
    if username == 'admin' and password == 'admin123':
        session['is_admin'] = True
        flash("Admin login successful!", "success")
        return redirect(url_for('admin_view'))

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            query = "SELECT * FROM users WHERE name = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()

        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['user_id']
            session['name'] = user['name']
            flash("Login successful!", "success")
            return redirect(url_for('language_home'))
        else:
            flash("Invalid username or password", "danger")
            return render_template('reg.html')

    except pymysql.MySQLError as err:
        flash(f"Database error: {err}", "danger")
        return render_template('reg.html')


@app.route('/language', methods=['GET', 'POST'])
def language_home():
    if request.method == 'POST':
        language = request.form.get('language')
        if language != None:
            return redirect(url_for('input_text', lang=language))
    return render_template('index.html')

@app.route('/input', methods=['GET', 'POST'])
def input_text():
    global original_input_text

    lang_code = request.args.get('lang')
    error = None
    result = None
    translation = None
    language_match = False
    translated_lang = None  # Variable to store the language of the translated text
    best_model = None  # Variable to store the best model suggestion

    if request.method == 'POST':
        input_mode = request.form.get('input_mode')

        if input_mode == 'text':
            sentence = request.form.get('typed_sentence', '').strip()
            if sentence:
                original_input_text = sentence
                detected_lang, confidence = detect_language(sentence)
                if detected_lang == lang_code:
                    result = "✅ Language match successful!"
                    language_match = True
                    high_resource_lang = find_high_resource_match(lang_code)
                    if high_resource_lang:
                        try:
                            translation = GoogleTranslator(source=lang_code, target=high_resource_lang).translate(sentence)
                            translated_lang = high_resource_lang  # Store the target language
                            best_model = f"The best model to use is {high_resource_lang.capitalize()} Model."
                        except Exception as e:
                            translation = f"Translation error: {e}"
                    else:
                        translation = "No suitable high-resource language found."
                else:
                    result = f"❌ Language mismatch! Detected: {detected_lang}, Expected: {lang_code.upper()}"
            else:
                error = "Please enter a non-empty sentence."

        elif input_mode == 'file':
            uploaded_file = request.files['text_file']
            if uploaded_file and uploaded_file.filename.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                lines = content.strip().splitlines()
                for line in lines:
                    if line:
                        original_input_text = line
                        detected_lang, confidence = detect_language(line)
                        if detected_lang == lang_code:
                            result = "✅ Language match successful!"
                            language_match = True
                            high_resource_lang = find_high_resource_match(lang_code)
                            if high_resource_lang:
                                try:
                                    translation = GoogleTranslator(source=lang_code, target=high_resource_lang).translate(line)
                                    translated_lang = high_resource_lang  # Store the target language
                                    best_model = f"The best model to use is {high_resource_lang.capitalize()} Model."
                                except Exception as e:
                                    translation = f"Translation error: {e}"
                            else:
                                translation = "No suitable high-resource language found."
                        else:
                            result = f"❌ Language mismatch! Detected: {detected_lang}, Expected: {lang_code.upper()}"
                        break
                if not result:
                    error = "Uploaded file contains no valid text."
            else:
                error = "Only .txt files are allowed. Please upload a valid text file."

    # Get the full language name
    translated_lang_name = LANGUAGES.get(translated_lang, translated_lang)  # Fallback to language code if not found

    return render_template('input_page.html', lang_code=lang_code, error=error, result=result,
                        translation=translation, language_match=language_match, 
                        translated_lang=translated_lang_name, best_model=best_model)
@app.route('/english')
def english_model():
    try:
        translated = GoogleTranslator(target='en').translate(original_input_text)
        cleaned = clean_text(translated)
        inputs = tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        sentiment = "Positive" if pred == 1 else "Negative"
        sentiment_score = probs[0][pred].item()

        # Log the result
        if 'user_id' in session:
            log_sentiment_result(session['user_id'], original_input_text, translated, sentiment, sentiment_score)

    except Exception as e:
        print(f"Translation or Sentiment Analysis error (English): {e}")
        sentiment = "Error"
        sentiment_score = 0.0

    return render_template('result_page.html', original_input=original_input_text, sentiment=sentiment, sentiment_score=sentiment_score)


@app.route('/spanish')
def spanish_model_route():
    try:
        # Translate the input to Spanish
        translated = GoogleTranslator(target='es').translate(original_input_text)
        print(f"Original Input (Spanish model): {original_input_text}")
        print(f"Translated to Spanish: {translated}")

        # Clean the translated text
        def clean_text(text):
            text = text.lower()
            text = re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        cleaned = clean_text(translated)

        # Load tokenizer and model if not already loaded
        model_path = './spanish_sentiment_model'
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        # Tokenize and predict
        inputs = tokenizer(cleaned, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        # Label mapping for 3-class model
        label_reverse_map = {0: "Negativo", 1: "Neutral", 2: "Positivo"}
        sentiment = label_reverse_map[pred]
        sentiment_score = probs[0][pred].item()

        print(f"Predicted Sentiment (Spanish): {sentiment}")
        print(f"Sentiment Score (Spanish): {sentiment_score}")

        # Log the result if user is logged in
        if 'user_id' in session:
            log_sentiment_result(session['user_id'], original_input_text, translated, sentiment, sentiment_score)

    except Exception as e:
        print(f"Translation or Sentiment Analysis error (Spanish): {e}")
        sentiment = "Error"
        sentiment_score = 0.0

    return render_template(
        'result_page.html',
        original_input=original_input_text,
        sentiment=sentiment,
        sentiment_score=sentiment_score
    )

@app.route('/french')
def french_model_route():
    try:
        # Translate the input to French
        translated = GoogleTranslator(target='fr').translate(original_input_text)
        print(f"Original Input (French model): {original_input_text}")
        print(f"Translated to French: {translated}")

        # Clean and tokenize the translated text
        cleaned = clean_text(translated)
        inputs = french_tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)

        # Perform sentiment analysis
        with torch.no_grad():
            outputs = french_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        sentiment = "Positive" if pred == 1 else "Negative"
        sentiment_score = probs[0][pred].item()

        print(f"Predicted Sentiment (French): {sentiment}")
        print(f"Sentiment Score (French): {sentiment_score}")

        # Log the result if the user is logged in
        if 'user_id' in session:
            log_sentiment_result(session['user_id'], original_input_text, translated, sentiment, sentiment_score)

    except Exception as e:
        print(f"Translation or Sentiment Analysis error (French): {e}")
        sentiment = "Error"
        sentiment_score = 0.0

    return render_template(
        'result_page.html',
        original_input=original_input_text,
        sentiment=sentiment,
        sentiment_score=sentiment_score
    )

@app.route('/hindi')
def hindi_model_route():
    try:
        # Translate the original input into Hindi
        translated = GoogleTranslator(target='hi').translate(original_input_text)
        print(f"Original Input (Hindi model): {original_input_text}")
        print(f"Translated to Hindi: {translated}")

        # Clean and tokenize the translated text
        cleaned = clean_text(translated)
        inputs = hindi_tokenizer(cleaned, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)

        # Predict sentiment
        with torch.no_grad():
            outputs = hindi_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        sentiment = "Negative" if pred == 1 else "Positive"
        sentiment_score = probs[0][pred].item()

        print(f"Predicted Sentiment (Hindi): {sentiment}")
        print(f"Sentiment Score (Hindi): {sentiment_score}")

        # Log the result if the user is logged in
        if 'user_id' in session:
            log_sentiment_result(session['user_id'], original_input_text, translated, sentiment, sentiment_score)

    except Exception as e:
        print(f"Translation or Sentiment Analysis error (Hindi): {e}")
        sentiment = "Error"
        sentiment_score = 0.0

    return render_template(
        'result_page.html',
        original_input=original_input_text,
        sentiment=sentiment,
        sentiment_score=sentiment_score
    )

@app.route('/admin')
def admin_view():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT u.name, u.email, sl.input_sentence, sl.translated_sentence, sl.sentiment, sl.sentiment_score, sl.timestamp FROM sentiment_logs sl JOIN users u ON sl.user_id = u.user_id ORDER BY sl.timestamp DESC")
            logs = cur.fetchall()
    except Exception as e:
        flash(f"Error fetching logs: {e}", "danger")
        logs = []
    finally:
        if conn:
            conn.close()
    return render_template('admin_portal.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True)
