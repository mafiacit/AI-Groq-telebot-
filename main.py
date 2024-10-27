import os
import base64
import mimetypes
import logging
import traceback
import tempfile
from collections import defaultdict
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

# Define vision-capable models
VISION_MODELS = [
    "llava-v1.5-7b-4096-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview"
]

# Define available models
MODELS = {
    "Llama 3.1 70B": "llama-3.1-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Llama 3.2 11B Vision": "llama-3.2-11b-vision-preview",
    "Llama 3.2 90B": "llama-3.2-90b-vision-preview",
    "LLaVA 1.5 7B": "llava-v1.5-7b-4096-preview",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
}

# Define bot modes
MODES = {
    "chat": "💬 General Chat",
    "document": "📄 Document Analysis",
    "image": "🖼️ Image Analysis",
    "code": "💻 Code Generation"
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
BUTTON_IMAGE_PATH = "button_layout.png"

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
        await update.message.reply_text("Please contact admin @kingkonfidents for access.")
        return

    # Send the button image using the cached version
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
    else:
        await query.edit_message_text(f"Mode set to: {MODES[mode]}. How can I help you?")

async def show_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = create_button_layout({k: f"🤖 {k}" for k in MODELS.keys()}, "select_model_")
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="back_to_mode")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.edit_text("Select a model:", reply_markup=reply_markup)

async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    model_name = query.data.split("_", 2)[2]
    user_id = update.effective_user.id
    user_preferences[user_id]["model"] = MODELS[model_name]
    await query.edit_message_text(f"Model set to: {model_name}. How can I help you?")

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

def get_groq_completion(model: str, messages: list, temperature: float, max_tokens: int):
    return groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Contact admin @kingkonfidents for access.")
        return

    user_id = update.effective_user.id
    user_message = update.message.text
    current_model = user_preferences[user_id]["model"]
    current_mode = user_preferences[user_id]["mode"]

    try:
        if current_mode == "image" and last_image[user_id]:
            image_data = last_image[user_id]
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image_data['mime_type']};base64,{image_data['base64']}"
                    }}
                ]}
            ]
        elif current_mode == "code":
            language = user_preferences[user_id]["language"]
            messages = [
                {"role": "system", "content": f"You are a helpful coding assistant. Generate {language.capitalize()} code."},
                {"role": "user", "content": f"Generate {language.capitalize()} code for: {user_message}"}
            ]
        else:
            messages = [{"role": "user", "content": user_message}]

        completion = get_groq_completion(current_model, messages, 0.7, 2000)

        if current_mode == "code":
            generated_code = completion.choices[0].message.content
            file_extension = LANGUAGES[user_preferences[user_id]["language"]]["extension"]
            with tempfile.NamedTemporaryFile(mode='w+', suffix=file_extension, delete=False) as temp_file:
                temp_file.write(generated_code)
                temp_filename = temp_file.name

            with open(temp_filename, "rb") as file:
                await update.message.reply_document(document=file, filename=f"generated_code{file_extension}")

            os.unlink(temp_filename)
            await update.message.reply_text(f"The generated {LANGUAGES[user_preferences[user_id]['language']]['emoji']} {user_preferences[user_id]['language'].capitalize()} code has been sent as a file.")
        else:
            await update.message.reply_text(completion.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        logger.error(traceback.format_exc())
        await update.message.reply_text("An error occurred. Please try again or rephrase your request.")

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

def main():
    # Preload the button image into cache
    load_button_image()

    # Start Flask server
    server_thread = Thread(target=run)
    server_thread.start()

    # Initialize bot
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(mode_callback, pattern="^mode_"))
    application.add_handler(CallbackQueryHandler(show_model_selection, pattern="^select_model$"))
    application.add_handler(CallbackQueryHandler(model_callback, pattern="^select_model_"))
    application.add_handler(CallbackQueryHandler(language_callback, pattern="^lang_"))
    application.add_handler(CallbackQueryHandler(show_mode_selection, pattern="^back_to_mode$"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handle_document))

    # Start bot
    logger.info("Bot is ready!")
    application.run_polling()

if __name__ == "__main__":
    main()
