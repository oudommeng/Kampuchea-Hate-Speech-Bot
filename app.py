import logging
import re
import nltk
from khmernltk import word_tokenize
import json
import joblib
from flask import Flask, request
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv
import os
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app_logs.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv("token.env")
TOKEN = os.getenv("TOKEN")
if not TOKEN:
    logger.error("No token found in environment variables!")
    exit(1)

# Download NLTK resources
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

# Load the saved models and vectorizer
try:
    MNB = joblib.load("model_output/mnb_model.pkl")
    vectorizer = joblib.load("model_output/vectorizer.pkl")
    logger.info(
        f"Model sizes: mnb_model.pkl={os.path.getsize('model_output/mnb_model.pkl')/1024:.2f}KB, "
        f"vectorizer.pkl={os.path.getsize('model_output/vectorizer.pkl')/1024:.2f}KB"
    )
except Exception as e:
    logger.error(f"Error loading ML models: {e}")
    exit(1)

# Helper functions for Khmer text preprocessing
def load_merge_map(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Error loading merge map: {e}")
        return {}

def remove_punc(text):
    return text

def remove_stopword(text):
    return text

def merge_word(cmt):
    merge_map = load_merge_map("merge_map.json")
    for phrase, merged in merge_map.items():
        escaped_phrase = re.escape(phrase).replace(r"\ ", r"\s+")
        regex_pattern = rf"(?<!\S){escaped_phrase}(?!\S)"
        cmt = re.sub(regex_pattern, merged, cmt)
    return cmt

def tokenize(cmt):
    try:
        words = word_tokenize(cmt, return_tokens=True)
        sentence = " ".join(word for word in words if word.strip())
        sentence = merge_word(sentence)
        sentence = remove_stopword(sentence)
        return sentence
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        return cmt

def generate_unigram(cmt):
    cmt = remove_punc(cmt)
    cmt = merge_word(cmt)
    words = word_tokenize(cmt, return_tokens=True)
    words = [word for word in words if word.strip()]
    return words

def generate_bigrams(words, n):
    return list(nltk.ngrams(words, n))

# Command handlers must be defined before create_application()
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to the Hate Speech Detection Bot! Send me a word, and I'll predict if it's hate speech or not."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    input_comment = update.message.text
    logger.info(f"Input received: {input_comment}")

    processed_comment = tokenize(input_comment)
    unigrams = generate_unigram(input_comment)
    bigrams = generate_bigrams(unigrams, 2)

    logger.info(f"Processed comment: {processed_comment}")

    try:
        comment_vec = vectorizer.transform([processed_comment])
        mnb_pred = MNB.predict(comment_vec)
        mnb_prediction = "Hate Speech" if mnb_pred[0] == 1 else "Non-Hate Speech"
        logger.info(f"Predictions - MNB: {mnb_prediction}")
        response = f"Prediction: {mnb_prediction}"
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        await update.message.reply_text("Sorry, an error occurred while processing your message.")

# Initialize Telegram application
def create_application():
    try:
        application = Application.builder().token(TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )
        
        return application
    except Exception as e:
        logger.error(f"Error creating Telegram application: {e}")
        return None

# Create application instance
application = create_application()
if not application:
    exit(1)

# Webhook route for Telegram
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    try:
        update = Update.de_json(request.get_json(), application.bot)
        application.update_queue.put(update)
        return "ok", 200
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return "error", 500

# Root route for health check
@app.route("/")
def index():
    return "Kampuchea Hate Speech Bot is running!"

if __name__ == "__main__":
    logger.info("Application started")
    
    # Check if we should use webhook (production) or polling (local)
    if os.getenv('USE_WEBHOOK', '').lower() in ('true', '1', 't'):
        # Webhook mode
        async def set_webhook():
            webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', 'your-render-app.onrender.com')}/{TOKEN}"
            await application.bot.setWebhook(webhook_url)
            logger.info(f"Webhook set to: {webhook_url}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(set_webhook())
        
        # Start Flask app
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    else:
        # Polling mode
        logger.info("Running in local polling mode")
        application.run_polling()