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
from threading import Thread

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
load_dotenv(".env")
TOKEN = os.getenv("TOKEN")
RENDER_EXTERNAL_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME")
USE_WEBHOOK = os.getenv("USE_WEBHOOK", "false").lower() in ("true", "1", "t")

if not TOKEN:
    logger.error("No TOKEN found in environment variables! Please set the Telegram bot token.")
    exit(1)

if USE_WEBHOOK and not RENDER_EXTERNAL_HOSTNAME:
    logger.error("No RENDER_EXTERNAL_HOSTNAME found in environment variables! Required for webhook mode.")
    exit(1)

logger.info(f"Environment variables - TOKEN: {'<set>' if TOKEN else '<not set>'}, "
            f"RENDER_EXTERNAL_HOSTNAME: {RENDER_EXTERNAL_HOSTNAME}, USE_WEBHOOK: {USE_WEBHOOK}")

# Download NLTK resources
try:
    nltk.download("punkt", quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

# Load the saved models and vectorizer
try:
    if not os.path.exists("model_output/mnb_model.pkl") or not os.path.exists("model_output/vectorizer.pkl"):
        raise FileNotFoundError("Model or vectorizer file not found in model_output directory")
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
        logger.info(f"Checking if {file_path} exists: {os.path.exists(file_path)}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")
        with open(file_path, "r", encoding="utf-8") as file:
            merge_map = json.load(file)
            logger.info(f"Merge map loaded successfully with {len(merge_map)} entries")
            return merge_map
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
        logger.info(f"Tokenized comment: {sentence}")
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

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("Executing start command")
    try:
        await update.message.reply_text(
            "Welcome to the Kampuchea Hate Speech Detection Bot! Send me a message, and I'll predict if it's hate speech or not."
        )
        logger.info("Start command response sent")
    except Exception as e:
        logger.error(f"Error in start command: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Handling message: {update.message.text}")
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
        logger.info("Message response sent")
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        await update.message.reply_text("Sorry, an error occurred while processing your message.")

# Initialize Telegram application
def create_application():
    try:
        application = Application.builder().token(TOKEN).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )
        logger.info("Telegram application created successfully")
        return application
    except Exception as e:
        logger.error(f"Error creating Telegram application: {e}")
        return None

# Create application instance
application = create_application()
if not application:
    logger.error("Failed to create Telegram application, exiting")
    exit(1)

# Test bot authentication
async def test_bot_token():
    try:
        bot_info = await application.bot.get_me()
        logger.info(f"Bot authentication successful: {bot_info.username} (ID: {bot_info.id})")
        return True
    except Exception as e:
        logger.error(f"Bot authentication failed: {e}")
        exit(1)

# Webhook route for Telegram
@app.route(f"/{TOKEN}", methods=["POST"])
async def webhook():
    try:
        update = Update.de_json(request.get_json(), application.bot)
        logger.info(f"Received update: {update}")
        if update:
            await application.process_update(update)
            logger.info("Update processed and added to queue")
            return "ok", 200
        else:
            logger.warning("Received empty or invalid update")
            return "ok", 200
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return "error", 500

# Root route for health check
@app.route("/")
def index():
    return "Kampuchea Hate Speech Bot is running!"

# Run application in a separate thread
def run_application():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(application.run_polling())
    except Exception as e:
        logger.error(f"Error running application: {e}")
    finally:
        loop.close()

if __name__ == "__main__":
    logger.info("Application started")
    
    if USE_WEBHOOK:
        logger.info("Entering webhook mode")
        # Webhook mode for Render
        async def start_application():
            try:
                await application.initialize()
                logger.info("Application initialized")
                await test_bot_token()  # Test token before setting webhook
                webhook_url = f"https://{RENDER_EXTERNAL_HOSTNAME}/{TOKEN}"
                logger.info(f"Attempting to set webhook: {webhook_url}")
                result = await application.bot.setWebhook(webhook_url)
                logger.info(f"Webhook set to: {webhook_url}, Success: {result}")
                if not result:
                    logger.error("Failed to set webhook")
                    exit(1)
            except Exception as e:
                logger.error(f"Error setting up webhook: {e}")
                exit(1)

        # Set up webhook
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(start_application())
        except Exception as e:
            logger.error(f"Error running event loop: {e}")
            exit(1)
        finally:
            loop.close()
        
        # Start Flask app
        logger.info("Starting Flask app")
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    else:
        # Polling mode for local development
        logger.info("Running in local polling mode")
        thread = Thread(target=run_application)
        thread.start()
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))