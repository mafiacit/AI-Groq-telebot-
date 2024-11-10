import os
import base64
import mimetypes
import logging
import traceback
import tempfile
import time
import aiohttp
import json
from collections import defaultdict, deque
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from groq import Groq
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

# Store user feedback with timestamp and feedback type
user_feedback = defaultdict(lambda: {})

# Store group conversation states
group_conversation_states = {}

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

# Define bot modes
MODES = {
    "chat": "💬 General Chat",
    "document": "📄 Document Analysis",
    "image": "🖼️ Image Analysis",
    "code": "💻 Code Generation"
}

# Define programming languages
LANGUAGES = {
    "python": {"emoji": "🐍", "extension": ".py"},
    "javascript": {"emoji": "🟨", "extension": ".js"},
    "java": {"emoji": "☕", "extension": ".java"},
    "cpp": {"emoji": "🔧", "extension": ".cpp"},
    "csharp": {"emoji": "🔷", "extension": ".cs"},
    "ruby": {"emoji": "💎", "extension": ".rb"},
    "go": {"emoji": "🔹", "extension": ".go"},
    "rust": {"emoji": "🦀", "extension": ".rs"},
    "swift": {"emoji": "🔶", "extension": ".swift"},
    "kotlin": {"emoji": "🟣", "extension": ".kt"},
    "php": {"emoji": "🐘", "extension": ".php"},
    "typescript": {"emoji": "🔵", "extension": ".ts"},
}

# Store user preferences
user_preferences = defaultdict(lambda: {"model": "llama-3.1-70b-versatile", "mode": "chat", "language": "python"})

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

# Rate limiting trackers
request_counts = defaultdict(lambda: {"minute": 0, "day": 0, "last_reset": time.time()})
token_counts = defaultdict(lambda: {"minute": 0, "day": 0, "last_reset": time.time()})

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

def get_groq_completion(model: str, messages: list, temperature: float, max_tokens: int):
    if not check_rate_limit(model, max_tokens):
        raise Exception("Rate limit exceeded")
    return groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

def is_user_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_USER_IDS

def create_button_layout(items, callback_prefix):
    return [[InlineKeyboardButton(text, callback_data=f"{callback_prefix}{key}")] for key, text in items.items()]

async def add_feedback_buttons(message, response_id: str):
    """Add feedback buttons to a message"""
    keyboard = [
        [
            InlineKeyboardButton("👍", callback_data=f"feedback_{response_id}_positive"),
            InlineKeyboardButton("👎", callback_data=f"feedback_{response_id}_negative")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    try:
        await message.edit_reply_markup(reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error adding feedback buttons: {str(e)}")
        try:
            await message.reply_text("Was this response helpful?", reply_markup=reply_markup)
        except Exception as e:
            logger.error(f"Error sending feedback message: {str(e)}")

async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle feedback button callbacks"""
    query = update.callback_query
    try:
        await query.answer()
        
        # Parse the callback data
        _, response_id, feedback_type = query.data.split("_")
        user_id = update.effective_user.id
        
        # Store the feedback with timestamp
        if response_id not in user_feedback[user_id]:
            user_feedback[user_id][response_id] = {
                'type': feedback_type,
                'timestamp': time.time()
            }
            
            # Update the message to show feedback received
            feedback_text = "👍 Thanks for positive feedback!" if feedback_type == "positive" else "👎 Thanks for negative feedback!"
            new_keyboard = [[InlineKeyboardButton(feedback_text, callback_data="feedback_received")]]
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(new_keyboard))
            
            # Log the feedback
            logger.info(f"User {user_id} gave {feedback_type} feedback for response {response_id}")
            
    except Exception as e:
        logger.error(f"Error handling feedback: {str(e)}")
        await query.edit_message_reply_markup(reply_markup=None)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_user_allowed(update.effective_user.id):
        await update.message.reply_text("Please contact admin @kingkonfidents for access.")
        return

    try:
        # Send welcome image
        with open('welcome.png', 'rb') as welcome_image:
            await update.message.reply_photo(
                photo=welcome_image,
                caption="Welcome to the AI Assistant! 🤖\n\n"
                "I can help you with:\n"
                "💬 General Chat\n"
                "📄 Document Analysis\n"
                "🖼️ Image Analysis\n"
                "💻 Code Generation\n\n"
                "Please select a mode to begin:"
            )
        
        # Show mode selection buttons
        keyboard = create_button_layout(MODES, "mode_")
        keyboard.append([InlineKeyboardButton("🤖 Select Model", callback_data="select_model")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text("Select a mode:", reply_markup=reply_markup)

    except FileNotFoundError:
        logger.error("Welcome image not found")
        await show_mode_selection(update, context, "Welcome! Please select a mode to begin:")
    except Exception as e:
        logger.error(f"Error in start command: {str(e)}")
        await show_mode_selection(update, context, "Welcome! Please select a mode to begin:")
        
async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enable AI conversation in the group"""
    if not update.message.chat.type in ['group', 'supergroup']:
        await update.message.reply_text("This command can only be used in groups.")
        return

    chat_id = update.message.chat_id
    user = update.effective_user

    if not is_user_allowed(user.id):
        await update.message.reply_text("You don't have permission to enable AI conversation.")
        return

    group_conversation_states[chat_id] = True
    await update.message.reply_text("AI conversation has been enabled in this group.\nUse /stop to disable it.")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Disable AI conversation in the group"""
    if not update.message.chat.type in ['group', 'supergroup']:
        await update.message.reply_text("This command can only be used in groups.")
        return

    chat_id = update.message.chat_id
    user = update.effective_user

    if not is_user_allowed(user.id):
        await update.message.reply_text("You don't have permission to disable AI conversation.")
        return

    group_conversation_states[chat_id] = False
    await update.message.reply_text("AI conversation has been disabled in this group.\nUse /ai to enable it again.")

async def show_mode_selection(update: Update, context: ContextTypes.DEFAULT_TYPE, custom_message: str = None):
    keyboard = create_button_layout(MODES, "mode_")
    keyboard.append([InlineKeyboardButton("🤖 Select Model", callback_data="select_model")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = custom_message or "Please select a mode:"
    
    if isinstance(update, Update):
        if update.message:
            await update.message.reply_text(message, reply_markup=reply_markup)
        else:
            await update.callback_query.message.edit_text(message, reply_markup=reply_markup)
    else:
        await update.message.edit_text(message, reply_markup=reply_markup)

async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mode = query.data.split("_")[1]
    user_id = update.effective_user.id
    user_preferences[user_id]["mode"] = mode
    
    if mode == "code":
        user_preferences[user_id]["language"] = "python"  # Default to Python
        await query.edit_message_text("Code generation mode activated. Send your coding request and I'll generate the code directly in a proper file.")
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.message.chat_id
    
    # Check if message is from a group
    if update.message.chat.type in ['group', 'supergroup']:
        # If AI is not enabled in this group, ignore the message
        if not group_conversation_states.get(chat_id, False):
            return
    
    if not is_user_allowed(user_id):
        await update.message.reply_text("Contact admin @kingkonfidents for access.")
        return

    user_message = update.message.text
    current_model = user_preferences[user_id]["model"]
    current_mode = user_preferences[user_id]["mode"]

    conversation_history[user_id].append({"role": "user", "content": user_message})

    try:
        if current_mode == "image" and last_image[user_id]:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{last_image[user_id]['base64']}"
                    }}
                ]}
            ]
        elif current_mode == "code":
            system_message = {
                "role": "system", 
                "content": f"You are a coding assistant. Generate clean, well-documented code. Include only the code without any markdown or explanation."
            }
            messages = [system_message] + list(conversation_history[user_id])
        else:
            messages = list(conversation_history[user_id])

        if current_mode != "image" and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": "You are a helpful AI assistant. Respond based on the conversation history."})

        completion = get_groq_completion(current_model, messages, 0.7, 2000)
        bot_response = completion.choices[0].message.content

        conversation_history[user_id].append({"role": "assistant", "content": bot_response})

        response_id = f"{user_id}_{int(time.time())}"

        if current_mode == "code":
            file_extension = LANGUAGES[user_preferences[user_id]["language"]]["extension"]
            safe_filename = "".join(x for x in user_message[:30] if x.isalnum() or x in (' ', '_')).strip()
            safe_filename = safe_filename.replace(' ', '_').lower()
            if not safe_filename:
                safe_filename = "generated_code"
            
            filename = f"{safe_filename}{file_extension}"

            with open(filename, "w", encoding='utf-8') as file:
                file.write(bot_response)

            sent_message = await update.message.reply_document(
                document=open(filename, "rb"),
                filename=filename,
                caption="Here's your generated code file."
            )
            os.remove(filename)
        else:
            if len(bot_response) > 4000:
                filename = "response.txt"
                with open(filename, "w", encoding='utf-8') as file:
                    file.write(bot_response)
                
                sent_message = await update.message.reply_document(
                    document=open(filename, "rb"),
                    filename=filename,
                    caption="The response was too long, so I've saved it to a file."
                )
                os.remove(filename)
            else:
                sent_message = await update.message.reply_text(bot_response)

        # Add feedback buttons to the sent message
        await add_feedback_buttons(sent_message, response_id)

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
                        "content": f"Please analyze this document (file: {file_name}):\n\n{truncated_content}"
                    }
                ],
                0.7,
                2000
            )

            bot_response = completion.choices[0].message.content
            response_id = f"{user_id}_{int(time.time())}"

            if len(bot_response) > 4000:
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(bot_response)
                    temp_filename = temp_file.name

                sent_message = await update.message.reply_document(
                    document=open(temp_filename, "rb"),
                    filename="analysis.txt",
                    caption="Document analysis (saved to file due to length)"
                )
                os.remove(temp_filename)
            else:
                sent_message = await update.message.reply_text(bot_response)

            await add_feedback_buttons(sent_message, response_id)

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        await update.message.reply_text("Sorry, there was an error processing your file. Please try again later.")

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_history[user_id].clear()
    await update.message.reply_text("Conversation history has been cleared.")

@app.route('/')
def home():
    return "Hello, I'm a Telegram AI!"

def run():
    app.run(host='0.0.0.0', port=8080)

def main():
    # Start the Flask server in a separate thread
    server_thread = Thread(target=run)
    server_thread.start()

    # Initialize the bot
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("ai", ai_command))
    application.add_handler(CommandHandler("stop", stop_command))
    
    # Add callback query handlers
    application.add_handler(CallbackQueryHandler(mode_callback, pattern="^mode_"))
    application.add_handler(CallbackQueryHandler(show_model_selection, pattern="^select_model$"))
    application.add_handler(CallbackQueryHandler(model_callback, pattern="^select_model_"))
    application.add_handler(CallbackQueryHandler(handle_feedback, pattern="^feedback_"))
    application.add_handler(CallbackQueryHandler(show_mode_selection, pattern="^back_to_mode$"))
    
    # Add message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handle_document))

    # Start the bot
    logger.info("AI is ready!")
    application.run_polling()

if __name__ == "__main__":
    main()      
