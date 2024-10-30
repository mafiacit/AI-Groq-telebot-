import os
import base64
import mimetypes
import logging
import traceback
import tempfile
import time
from collections import defaultdict, deque
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from groq import Groq
import aiohttp
from flask import Flask
from threading import Thread

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Get allowed user IDs from environment variable
ALLOWED_USER_IDS = set(map(int, os.getenv("ALLOWED_USER_IDS", "").split(',')))

# Store the last image for each user
last_image = defaultdict(lambda: None)

# Store conversation history for each user
conversation_history = defaultdict(lambda: deque(maxlen=10))

# Define all available models
MODELS = {
    "Gemma 7B": "gemma-7b-it",
    "Gemma 9B": "gemma2-9b-it",
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Llama 3.2 11B Text": "llama-3.2-11b-text-preview",
    "Llama 3.2 11B Vision": "llama-3.2-11b-vision-preview",
    "Llama 3.2 1B": "llama-3.2-1b-preview",
    "Llama 3.2 3B": "llama-3.2-3b-preview",
    "Llama 3.2 90B Text": "llama-3.2-90b-text-preview",
    "Llama 3.2 90B Vision": "llama-3.2-90b-vision-preview",
    "Llama Guard 3 8B": "llama-guard-3-8b",
    "Llama3 70B 8192": "llama3-70b-8192",
    "Llama3 8B 8192": "llama3-8b-8192",
    "Llama3 Groq 70B 8192": "llama3-groq-70b-8192-tool-use-preview",
    "Llama3 Groq 8B 8192": "llama3-groq-8b-8192-tool-use-preview",
    "LLaVA 1.5 7B": "llava-v1.5-7b-4096-preview",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
}

# Define speech-to-text models
STT_MODELS = {
    "Distil Whisper Large V3 EN": "distil-whisper-large-v3-en",
    "Whisper Large V3": "whisper-large-v3",
    "Whisper Large V3 Turbo": "whisper-large-v3-turbo",
}

# Define rate limits
RATE_LIMITS = {
    "gemma-7b-it": {"rpm": 30, "rpd": 14400, "tpm": 15000, "tpd": 500000},
    "gemma2-9b-it": {"rpm": 30, "rpd": 14400, "tpm": 15000, "tpd": 500000},
    "llama-3.1-70b-versatile": {"rpm": 30, "rpd": 14400, "tpm": 6000, "tpd": 200000},
    "llama-3.1-8b-instant": {"rpm": 30, "rpd": 14400, "tpm": 20000, "tpd": 500000},
    "llama-3.2-11b-text-preview": {"rpm": 30, "rpd": 7000, "tpm": 7000, "tpd": 500000},
    "llama-3.2-11b-vision-preview": {"rpm": 30, "rpd": 7000, "tpm": 7000, "tpd": 500000},
    "llama-3.2-1b-preview": {"rpm": 30, "rpd": 7000, "tpm": 7000, "tpd": 500000},
    "llama-3.2-3b-preview": {"rpm": 30, "rpd": 7000, "tpm": 7000, "tpd": 500000},
    "llama-3.2-90b-text-preview": {"rpm": 30, "rpd": 7000, "tpm": 7000, "tpd": 500000},
    "llama-3.2-90b-vision-preview": {"rpm": 15, "rpd": 3500, "tpm": 7000, "tpd": 250000},
    "llama-guard-3-8b": {"rpm": 30, "rpd": 14400, "tpm": 15000, "tpd": 500000},
    "llama3-70b-8192": {"rpm": 30, "rpd": 14400, "tpm": 6000, "tpd": 500000},
    "llama3-8b-8192": {"rpm": 30, "rpd": 14400, "tpm": 30000, "tpd": 500000},
    "llama3-groq-70b-8192-tool-use-preview": {"rpm": 30, "rpd": 14400, "tpm": 15000, "tpd": 500000},
    "llama3-groq-8b-8192-tool-use-preview": {"rpm": 30, "rpd": 14400, "tpm": 15000, "tpd": 500000},
    "llava-v1.5-7b-4096-preview": {"rpm": 30, "rpd": 14400, "tpm": 30000, "tpd": float('inf')},
    "mixtral-8x7b-32768": {"rpm": 30, "rpd": 14400, "tpm": 5000, "tpd": 500000},
}

STT_RATE_LIMITS = {
    "distil-whisper-large-v3-en": {"rpm": 20, "rpd": 2000, "sph": 7200, "spd": 28800},
    "whisper-large-v3": {"rpm": 20, "rpd": 2000, "sph": 7200, "spd": 28800},
    "whisper-large-v3-turbo": {"rpm": 20, "rpd": 2000, "sph": 7200, "spd": 28800},
}

# Rate limiting trackers
request_counts = defaultdict(lambda: {"minute": 0, "day": 0, "last_reset": time.time()})
token_counts = defaultdict(lambda: {"minute": 0, "day": 0, "last_reset": time.time()})
stt_request_counts = defaultdict(lambda: {"minute": 0, "day": 0, "last_reset": time.time()})
stt_audio_duration = defaultdict(lambda: {"hour": 0, "day": 0, "last_reset": time.time()})

# Define bot modes
MODES = {
    "chat": "💬 General Chat",
    "document": "📄 Document Analysis",
    "image": "🖼️ Image Analysis",
    "code": "💻 Code Generation",
    "speech": "🎤 Speech to Text"
}

# Define programming languages for code generation with their file extensions
LANGUAGES = {
    "python": {"emoji": "🐍", "extension": ".py"},
    "javascript": {"emoji": "🟨", "extension": ".js"},
    "java": {"emoji": "☕", "extension": ".java"},
    "cpp": {"emoji": "🔧", "extension": ".cpp"},
    "csharp": {"emoji": "🔷", "extension": ".cs"},
    "ruby": {"emoji": "💎", "extension": ".rb"},
    "go": {"emoji": "🐹", "extension": ".go"},
    "rust": {"emoji": "🦀", "extension": ".rs"},
    "swift": {"emoji": "🕊", "extension": ".swift"},
    "kotlin": {"emoji": "🟠", "extension": ".kt"},
    "php": {"emoji": "🐘", "extension": ".php"},
    "typescript": {"emoji": "🔵", "extension": ".ts"},
}

# Store user preferences
user_preferences = defaultdict(lambda: {"model": "llama-3.1-70b-versatile", "mode": "chat", "language": "python"})

# Path to the button image file
BUTTON_IMAGE_PATH = "IMG_3008.png"

# Cache for the button image
button_image_cache = None

def load_button_image():
    global button_image_cache
    if button_image_cache is None:
        if os.path.exists(BUTTON_IMAGE_PATH):
            with open(BUTTON_IMAGE_PATH, "rb") as image_file:
                button_image_cache = image_file.read()
        else:
            logger.error(f"Button image file not found at {BUTTON_IMAGE_PATH}")
    return button_image_cache

@app.route('/')
def home():
    return "Hello, I'm a Telegram bot!"

def run():
    app.run(host='0.0.0.0', port=8080)

def is_user_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USER_IDS

def create_button_layout(items, callback_prefix):
    return [[InlineKeyboardButton(text, callback_data=f"{callback_prefix}{key}")] for key, text in items.items()]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Please contact admin @anonymousboyzzs for access.")
        return

    button_image = load_button_image()
    if button_image:
        await update.message.reply_photo(button_image, caption="The Sentinel of Tomorrow: A Glimpse into AI Evolution")
    else:
        await update.message.reply_text("Welcome! (Button image not available)")

    await show_mode_selection(update, context)

async def show_mode_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = create_button_layout(MODES, "mode_")
    keyboard.append([InlineKeyboardButton("🤖 Select Model", callback_data="select_model")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    message = "Welcome! I'm here to assist you. Please select a mode or change the model:"
    if update.message:
        await update.message.reply_text(message, reply_markup=reply_markup)
    else:
        await update.callback_query.message.edit_text(message, reply_markup=reply_markup)

async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mode = query.data.split("_")[1]
    user_id = update.effective_user.id
    user_preferences[user_id]["mode"] = mode
    if mode == "code":
        await show_language_selection(update, context)
    elif mode == "image":
        user_preferences[user_id]["model"] = "llama-3.2-11b-vision-preview"
        await query.edit_message_text(f"Mode set to: {MODES[mode]}. Please send an image for analysis.")
    elif mode == "speech":
        await show_stt_model_selection(update, context)
    else:
        await query.edit_message_text(f"Mode set to: {MODES[mode]}. How can I help you?")

async def show_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = create_button_layout({k: f"🤖 {k}" for k in MODELS.keys()}, "select_model_")
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="back_to_mode")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.edit_text("Select a model:", reply_markup=reply_markup)

async def show_stt_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = create_button_layout({k: f"🎤 {k}" for k in STT_MODELS.keys()}, "select_stt_model_")
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="back_to_mode")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.edit_text("Select a speech-to-text model:", reply_markup=reply_markup)

async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    model_name = query.data.split("_", 2)[2]
    user_id = update.effective_user.id
    user_preferences[user_id]["model"] = MODELS[model_name]
    await query.edit_message_text(f"Model set to: {model_name}. How can I help you?")

async def stt_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    model_name = query.data.split("_", 3)[3]
    user_id = update.effective_user.id
    user_preferences[user_id]["stt_model"] = STT_MODELS[model_name]
    await query.edit_message_text(f"Speech-to-text model set to: {model_name}. Please send an audio message.")

async def show_language_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(f"{lang_info['emoji']} {lang.capitalize()}", callback_data=f"lang_{lang}")] for lang, lang_info in LANGUAGES.items()]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.edit_text("Select a programming language:", reply_markup=reply_markup)

async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang = query.data.split("_")[1]
    user_id = update.effective_user.id
    user_preferences[user_id]["language"] = lang
    await query.edit_message_text(f"Language set to: {LANGUAGES[lang]['emoji']} {lang.capitalize()}. Please provide your code generation request.")

def reset_rate_limits():
    current_time = time.time()
    for model, counts in request_counts.items():
        if current_time - counts["last_reset"] >= 60:
            counts["minute"] = 0
            counts["last_reset"] = current_time
        if current_time - counts["last_reset"] >= 86400:
            counts["day"] = 0
    for model, counts in token_counts.items():
        if current_time - counts["last_reset"] >= 60:
            counts["minute"] = 0
            counts["last_reset"] = current_time
        if current_time - counts["last_reset"] >= 86400:
            counts["day"] = 0
    for model, counts in stt_request_counts.items():
        if current_time - counts["last_reset"] >= 60:
            counts["minute"] = 0
            counts["last_reset"] = current_time
        if current_time - counts["last_reset"] >= 86400:
            counts["day"] = 0
    for model, counts in stt_audio_duration.items():
        if current_time - counts["last_reset"] >= 3600:
            counts["hour"] = 0
            counts["last_reset"] = current_time
        if current_time - counts["last_reset"] >= 86400:
            counts["day"] = 0

def check_rate_limit(model: str, tokens: int = 0):
    reset_rate_limits()
    limits = RATE_LIMITS.get(model, {})
    if not limits:
        return True
    
    request_counts[model]["minute"] += 1
    request_counts[model]["day"] += 1
    token_counts[model]["minute"] += tokens
    token_counts[model]["day"] += tokens

    return (request_counts[model]["minute"] <= limits["rpm"] and
            request_counts[model]["day"] <= limits["rpd"] and
            token_counts[model]["minute"] <= limits["tpm"] and
            token_counts[model]["day"] <= limits["tpd"])

def check_stt_rate_limit(model: str, audio_duration: float):
    reset_rate_limits()
    limits = STT_RATE_LIMITS.get(model, {})
    if not limits:
        return True
    
    stt_request_counts[model]["minute"] += 1
    stt_request_counts[model]["day"] += 1
    stt_audio_duration[model]["hour"] += audio_duration
    stt_audio_duration[model]["day"] += audio_duration

    return (stt_request_counts[model]["minute"] <= limits["rpm"] and
            stt_request_counts[model]["day"] <= limits["rpd"] and
            stt_audio_duration[model]["hour"] <= limits["sph"] and
            stt_audio_duration[model]["day"] <= limits["spd"])

def get_groq_completion(model: str, messages: list, temperature: float, max_tokens: int):
    if not check_rate_limit(model, max_tokens):
        raise Exception("Rate limit exceeded")
    return groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE, transcribed_text: str = None):
    if not is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Contact admin @kingkonfidents for access.")
        return

    user_id = update.effective_user.id
    user_message = transcribed_text or update.message.text
    current_model = user_preferences[user_id]["model"]
    current_mode = user_preferences[user_id]["mode"]

    conversation_history[user_id].append({"role": "user", "content": user_message})

    try:
        if current_mode == "image" and last_image[user_id]:
            image_data = last_image[user_id]
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data['base64']}"
                    }}
                ]}
            ]
        elif current_mode == "code":
            language = user_preferences[user_id]["language"]
            system_message = {"role": "system", "content": f"You are a helpful coding assistant. Generate {language.capitalize()} code."}
            messages = [system_message] + list(conversation_history[user_id])
        else:
            messages = list(conversation_history[user_id])

        if current_mode != "image" and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": "You are a helpful AI assistant. Respond based on the conversation history."})

        completion = get_groq_completion(current_model, messages, 0.7, 2000)
        bot_response = completion.choices[0].message.content

        conversation_history[user_id].append({"role": "assistant", "content": bot_response})

        if current_mode == "code":
            file_extension = LANGUAGES[user_preferences[user_id]["language"]]["extension"]
            with tempfile.NamedTemporaryFile(mode='w+', suffix=file_extension, delete=False) as temp_file:
                temp_file.write(bot_response)
                temp_filename = temp_file.name

            with open(temp_filename, "rb") as file:
                await update.message.reply_document(document=file, filename=f"generated_code{file_extension}")

            os.unlink(temp_filename)
            await update.message.reply_text(f"The generated {LANGUAGES[user_preferences[user_id]['language']]['emoji']} {user_preferences[user_id]['language'].capitalize()} code has been sent as a file.")
        else:
            await update.message.reply_text(bot_response)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        logger.error(traceback.format_exc())
        await update.message.reply_text("An error occurred. Please try again or rephrase your request.")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return

    user_id = update.effective_user.id
    current_stt_model = user_preferences[user_id].get("stt_model", STT_MODELS["Whisper Large V3"])

    if update.message.voice:
        file = await context.bot.get_file(update.message.voice.file_id)
        duration = update.message.voice.duration
    else:
        await update.message.reply_text("Please send a voice message.")
        return

    try:
        if not check_stt_rate_limit(current_stt_model, duration):
            await update.message.reply_text("Speech-to-text rate limit exceeded. Please try again later.")
            return

        # Download the voice message
        voice_file = await file.download_as_bytearray()

        # Here you would typically send the audio to a speech-to-text API
        # For now, we'll simulate the transcription
        transcribed_text = f"Simulated transcription for {duration} seconds of audio using {current_stt_model}"

        await update.message.reply_text(f"Transcription: {transcribed_text}")

        # Call handle_message with the transcribed text
        await handle_message(update, context, transcribed_text)

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        await update.message.reply_text("Sorry, there was an error processing your audio. Please try again later.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return

    user_id = update.effective_user.id

    if update.message.document:
        file = await context.bot.get_file(update.message.document.file_id)
        file_name = update.message.document.file_name
    elif update.message.photo:
        file = await context.bot.get_file(update.message.photo[-1].file_id)
        file_name = f"photo_{update.message.photo[-1].file_id}.jpg"
    else:
        await update.message.reply_text("Please send a document or image file.")
        return

    mime_type, _ = mimetypes.guess_type(file_name)

    try:
        if mime_type and mime_type.startswith('image'):
            async with aiohttp.ClientSession() as session:
                async with session.get(file.file_path) as resp:
                    file_content = await resp.read()

            base64_image = base64.b64encode(file_content).decode('utf-8')
            last_image[user_id] = {
                'base64': base64_image,
                'mime_type': mime_type
            }

            await update.message.reply_text("Image received. You can now ask me to analyze it.")
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(file.file_path) as resp:
                    file_content = await resp.text()

            max_chars = 15000
            truncated_content = file_content[:max_chars]

            current_model = user_preferences[user_id]["model"]
            completion = get_groq_completion(
                current_model,
                [
                    {
                        "role": "user",
                        "content": f"Please analyze and summarize the following document (file name: {file_name}):\n\n{truncated_content}"
                    }
                ],
                0.7,
                2000
            )

            bot_response = completion.choices[0].message.content

            if len(bot_response) > 4000:
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(bot_response)
                    temp_filename = temp_file.name

                with open(temp_filename, "rb") as file:
                    await update.message.reply_document(document=file, filename="document_analysis.txt")

                os.unlink(temp_filename)
                await update.message.reply_text("The document analysis was too long to send as a message. I've sent it as a text file instead.")
            else:
                await update.message.reply_text(f"Document analysis:\n\n{bot_response}")

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        await update.message.reply_text("Sorry, there was an error processing your file. Please try again later.")

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_history[user_id].clear()
    await update.message.reply_text("Conversation history has been cleared.")

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = conversation_history[user_id]
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    await update.message.reply_text(f"Current conversation history:\n\n{history_text}")

def main():
    load_button_image()

    server_thread = Thread(target=run)
    server_thread.start()

    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("history", show_history))
    application.add_handler(CallbackQueryHandler(mode_callback, pattern="^mode_"))
    application.add_handler(CallbackQueryHandler(show_model_selection, pattern="^select_model$"))
    application.add_handler(CallbackQueryHandler(model_callback, pattern="^select_model_"))
    application.add_handler(CallbackQueryHandler(stt_model_callback, pattern="^select_stt_model_"))
    application.add_handler(CallbackQueryHandler(language_callback, pattern="^lang_"))
    application.add_handler(CallbackQueryHandler(show_mode_selection, pattern="^back_to_mode$"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.VOICE, handle_audio))

    logger.info("Bot is ready!")
    application.run_polling()

if __name__ == "__main__":
    main()
