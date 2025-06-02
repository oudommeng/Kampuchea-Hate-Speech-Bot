import logging
import datetime
import re
import nltk
from khmernltk import word_tokenize
import json
import joblib
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import os
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_logs.log'),
        logging.StreamHandler()
    ]
)

load_dotenv('token.env')

logger = logging.getLogger(__name__)

# Load the saved models and vectorizer
MNB = joblib.load('model_output/mnb_model.pkl')
# BNB = joblib.load('model_output/bnb_model.pkl')
vectorizer = joblib.load('model_output/vectorizer.pkl')

# Helper functions for Khmer text preprocessing
def load_merge_map(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Error loading merge map: {e}")
        return {}

def remove_punc(text):
    # Placeholder for punctuation removal function
    # Add your punctuation removal logic here if needed
    return text

def remove_stopword(text):
    # Placeholder for stopword removal function
    # Add your stopword removal logic here if needed
    return text

def merge_word(cmt):
    merge_map = load_merge_map("merge_map.json")
    for phrase, merged in merge_map.items():
        escaped_phrase = re.escape(phrase).replace(r'\ ', r'\s+')
        regex_pattern = rf'(?<!\S){escaped_phrase}(?!\S)'
        cmt = re.sub(regex_pattern, merged, cmt)
    return cmt

def tokenize(cmt):
    words = word_tokenize(cmt, return_tokens=True)
    sentence = ' '.join(word for word in words if word.strip())
    sentence = merge_word(sentence)
    sentence = remove_stopword(sentence)
    return sentence

def generate_unigram(cmt):
    cmt = remove_punc(cmt)
    cmt = merge_word(cmt)
    words = word_tokenize(cmt, return_tokens=True)
    words = [word for word in words if word.strip()]
    return words

def generate_bigrams(words, n):
    return list(nltk.ngrams(words, n))


# Get token from environment variables
TOKEN = os.environ.get('TOKEN')
if not TOKEN:
    logger.error("No token found in environment variables!")
    exit(1)


# Handle the /start command
async def start(update: Update, context):
    await update.message.reply_text(
        "Welcome to the Hate Speech Detection Bot! Send me a comment, and I’ll predict if it’s hate speech or not."
    )

# Handle incoming text messages
async def handle_message(update: Update, context):
    input_comment = update.message.text
    logger.info(f"Input received: {input_comment}")

    # Process input with Khmer NLP
    processed_comment = tokenize(input_comment)
    unigrams = generate_unigram(input_comment)
    bigrams = generate_bigrams(unigrams, 2)

    logger.info(f"Processed comment: {processed_comment}")

    # Vectorize the processed comment
    comment_vec = vectorizer.transform([processed_comment])

    # Make predictions
    mnb_pred = MNB.predict(comment_vec)
    # bnb_pred = BNB.predict(comment_vec)

    # Convert predictions to readable labels
    mnb_prediction = "Hate Speech" if mnb_pred[0] == 1 else "Non-Hate Speech"
    # bnb_prediction = "Hate Speech" if bnb_pred[0] == 1 else "Non-Hate Speech"

    # Log the predictions
    logger.info(f"Predictions - MNB: {mnb_prediction}")

    # Send response to the user
    response = f"{mnb_prediction}"
    await update.message.reply_text(response)

if __name__ == '__main__':
    # Download necessary NLTK resources
    try:
        nltk.download('punkt')
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")

    logger.info("Application started")

    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()