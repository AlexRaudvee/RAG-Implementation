# Imports
import os
import logging

from config import model
from envvar import BOT_TOKEN
from functions.functions import *

from aiogram import F
from aiogram.filters import Command
from aiogram.types import BotCommand
from aiogram import Bot, Dispatcher, types, flags

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)  # Logs important information for debugging and monitoring

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)  # Initialize the bot with the provided token
dp = Dispatcher()  # Dispatcher to manage and handle events and updates

# Global Variables
MODE = None  # Mode of operation: 'chat_with_docs' or 'internet_search'
CONTEXT = []  # Conversation history for contextual replies
LNG = None  # Current language setting ('en' or 'ru')
CHUNKS = set()  # Unique document chunks used for context
CHUNKS_NUMBERS = []  # Track chunk numbers for efficient management

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector database
try:
    os.rmdir('db')  # Remove the database directory if it exists
except:
    pass

VECTOR_STORE = Chroma(
    collection_name="collection",  # Name of the vector database collection
    embedding_function=embeddings,  # Embedding model for semantic search
    persist_directory="./db",  # Directory to store the database
)

# Command Handlers
@dp.message(Command("language"))
async def set_language(message: types.Message):
    """
    Handle the /language command to let the user choose their preferred language.

    Args:
        message (types.Message): Telegram message object.
    """
    # Inline buttons for language selection
    buttons = [
        [
            types.InlineKeyboardButton(text="English", callback_data="en"),
            types.InlineKeyboardButton(text="Русский", callback_data="ru"),
        ],
    ]
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)

    # Send language selection message
    await message.answer(
        "Choose the language / Выберите язык:",
        reply_markup=keyboard
    )

@dp.callback_query(F.data == "en")
async def lang_change(callback: types.CallbackQuery):
    """
    Set the bot language to English when the user selects it.

    Args:
        callback (types.CallbackQuery): Callback query object triggered by the user.
    """
    global LNG
    LNG = 'en'
    await callback.answer()  # Acknowledge the callback
    await callback.message.answer("Language changed to English")

@dp.callback_query(F.data == "ru")
async def lang_change(callback: types.CallbackQuery):
    """
    Set the bot language to Russian when the user selects it.

    Args:
        callback (types.CallbackQuery): Callback query object triggered by the user.
    """
    global LNG
    LNG = 'ru'
    await callback.answer()  # Acknowledge the callback
    await callback.message.answer("Язык изменен на Русский")

# Handler for receiving PDF files
@dp.message(F.document)
async def handle_pdf_file(message: types.Message):
    """
    Handle uploaded PDF or TXT files, process them, and initialize context for conversation.

    Args:
        message (types.Message): Telegram message containing the file.
    """
    # Check if the uploaded file is a PDF or TXT
    if message.document.mime_type in ('application/pdf', 'application/txt'):
        # Send a sticker as feedback
        msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNCVnQjEYjAwGtQeYOD5qd97Qn8FiwgACkAIAAsbveUXqtGM6TksDCjYE")
        
        # Ensure 'documents' directory exists
        os.makedirs("documents", exist_ok=True)

        # Retrieve file information
        file_id = message.document.file_id
        file_name = message.document.file_name
        file_path = f"documents/{file_name}"

        # Download and save the file locally
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, file_path)

        # Process the document
        docs = upload_document(file_path)  # Custom function to upload and parse the document
        
        # Update global variables
        global CONTEXT, CHUNKS, MODE, VECTOR_STORE
        add_to_vector_db(vector_db=VECTOR_STORE, docs=docs)  # Add document to the vector database
        CONTEXT = []
        CHUNKS = set()
        MODE = "chat_with_docs"  # Set mode to document-based chat

        # Delete the sticker and send confirmation
        await bot.delete_message(msg.chat.id, msg.message_id)
        await message.answer("Okay, let's talk about this file.")
    else:
        # Notify the user if the file is not a PDF or TXT
        await message.answer("Please upload a valid PDF or TXT file.")

# Handler for internet search
@dp.message(Command("search"))
async def handle_search(message: types.Message):
    """
    Handle the /search command to activate the internet search mode.

    Args:
        message (types.Message): Telegram message object.
    """
    global CONTEXT, CHUNKS, MODE
    CONTEXT = []  # Reset context
    CHUNKS = set()  # Reset chunks
    MODE = "internet_search"  # Set mode to internet search

    # Inform the user about the activated mode
    await message.answer("This is a demo of *HAOSearch \\(Hunt All Over Search\\)*\\. So ask me anything and will look for that in the entire internet", parse_mode="MarkdownV2")

# General message handler
@dp.message(F.text & ~F.text.startswith('/'))
async def send_message(message: types.Message):
    """
    Handle user messages and generate a response based on the current mode (document chat or internet search).

    Args:
        message (types.Message): Telegram message containing text input.
    """
    global CONTEXT, CHUNKS, CHUNKS_NUMBERS, MODE, VECTOR_STORE

    if MODE == "chat_with_docs":
        # Respond with document-based context
        if len(CONTEXT) < 2:
            # Show typing feedback and retrieve context
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")
            CHUNKS = list({doc.page_content: doc for doc in retrieve_context(VECTOR_STORE, message.text, k=4)}.values())
            
            # Create a prompt and generate response
            full_context = "\n".join([doc.page_content for doc in CHUNKS])
            prompt = f"With the provided CONTEXT, answer the following question:\n\nCONTEXT:\n{full_context}\n\nQUESTION:\n{message.text}"
            response, _, _, _ = generate_conversation_with_context(model, CHUNKS, message.text, CONTEXT, prompt)

            # Save the conversation context
            CONTEXT.extend([
                {'role': 'user', 'parts': [prompt]},
                {'role': 'model', 'parts': [response]}
            ])
            
            # Send the response to the user
            response = transform_text(response)
            await bot.delete_message(msg.chat.id, msg.message_id)
            await message.answer(response, parse_mode="MarkdownV2")

    elif MODE == "internet_search":

        # Ensure the conversation history context is minimal for simplicity
        if len(CONTEXT) < 2:

            # Step 1: Notify the user about the search process by sending a sticker
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNBdnQh--e8ycq2zhVm-A5zMS4eumbwACRgMAAiqHGURoXzCXdu7QsTYE")

            # Use a custom function to perform the internet search and convert results to documents
            docs = search_info_to_docs(model, question=f"{message.text}")

            # Remove the initial sticker message once the task is done
            await bot.delete_message(msg.chat.id, msg.message_id)

            # Step 2: Notify the user about embedding the search results
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNCVnQjEYjAwGtQeYOD5qd97Qn8FiwgACkAIAAsbveUXqtGM6TksDCjYE")

            # Add the retrieved documents to the vector database for embedding and context retrieval
            add_to_vector_db(VECTOR_STORE, docs)

            # Retrieve the most relevant chunks of information from the vector database
            CHUNKS = retrieve_context(VECTOR_STORE, message.text, k=4)

            # Remove the second sticker after embedding is complete
            await bot.delete_message(msg.chat.id, msg.message_id)

            # Step 3: Notify the user that the response is being generated
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENLYZnPbu98m8InnQC4bXdrzcTronVogACeQMAAo13GUSF6cj_mt_hyjYE")

            # Combine the retrieved chunks to form a single context string
            full_context = "\n\n\n".join([doc.page_content for doc in CHUNKS])

            # Create a prompt for the model using the context and user question
            prompt = f"""Keeping provided CONTEXT in mind answer my question, your answer should be well structured and answer all aspects of my question 
                        \n\nCONTEXT:\n{full_context}\n\nQUESTION:\n{message.text}"""

            # Generate a response using a custom function with the model, context, and prompt
            response, role, query, chat = generate_conversation_with_context(
                model=model, 
                context=CHUNKS, 
                query=message.text, 
                history=CONTEXT, 
                prompt=prompt
            )

            # Step 4: Update the conversation context with the user question and model response
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})

            # Step 5: Extract and format resource links from document metadata
            links = []
            for doc in CHUNKS:
                try:
                    title = doc.metadata['title']  # Extract document title if available
                except:
                    title = "Unknown URL"  # Fallback title if metadata is missing
                
                source = doc.metadata['source']  # Extract document source (URL)
                links.append((source, title))  # Add to the list of links
            
            # Remove duplicate links and format them as a markdown list
            links = list(set(links))
            links = '\n'.join(f'• [{transform_text(link[1])}]({link[0]})' for link in links)

            # Transform the response text for better formatting
            response = transform_text(response)

            # Combine the response and resources into the final message
            text = f"{response}\n*Resources:*\n{links}"

            # Remove the sticker used during response generation
            await bot.delete_message(msg.chat.id, msg.message_id)

            # Send the final formatted response to the user
            await message.answer(text=text, parse_mode="MarkdownV2")

        elif len(CONTEXT) >= 2:
            
            # notify user about writing of the message by sending a sticker
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")
            prompt = f"{message.text}"
            
            # generate response
            response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=message.text, history=CONTEXT, prompt=prompt)
            
            # add everything context
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
            
            # transform text from plain to markdown
            response = transform_text(response)
            
            # delete the sticker
            await bot.delete_message(msg.chat.id, msg.message_id)
            # answer to the user
            await message.answer(f"{response}", parse_mode="MarkdownV2")



# Command: /start
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    """
    Send a welcome message and reset the bot to its initial state.

    Args:
        message (types.Message): Telegram message triggering the command.
    """
    global VECTOR_STORE, CONTEXT, CHUNKS
    recreate_directory("db")  # Clear the database directory
    VECTOR_STORE = Chroma(
        collection_name="collection",
        embedding_function=embeddings,
        persist_directory="db",
    )
    CONTEXT = []
    CHUNKS = set()

    # Select the introduction based on the language
    en_intro = ("*Welcome to InfoSphere Telegram Bot*\\. To start, you can send a PDF or TXT file, "
                "or use the following commands:\n/start \\- Restart the bot\\.\n/language \\- Change the language\\.\n/search \\- HAOSearch the internet\\.")
    ru_intro = ("*Добро пожаловать в CookBook Telegram Бот*\\. Вы можете загрузить PDF или TXT файл или использовать команды:\n/start \\- Запустить бота\\.\n/language \\- Сменить язык\\.\n/search \\- Искать ответ в интернете\\.")
    prompt = ru_intro if LNG == 'ru' else en_intro

    await message.answer(text=prompt, parse_mode="MarkdownV2")

# Command: /help 
@dp.message(Command("help"))
async def send_help(message: types.Message):
    """
    Send a help message
    """
    help_text = ("*Help*\\. \nYou can use the following commands:\n/start \\- This command restart the bot\\.\n/search \\- This command activates the HAOSearch mode, when the bot is able to look out online resources to answer your question\\.\n/help \\- Holp command which returns all the commands and instruction for this bot\\.")
    
    await message.answer(help_text, parse_mode="MarkdownV2")
    
# Main function
async def main():
    """
    Main entry point for starting the bot.
    """
    # Set bot commands
    await bot.set_my_commands([
        BotCommand(command="/search", description="Search the internet"),
        BotCommand(command="/language", description="Change language"),
        BotCommand(command="/start", description="Restart the bot"),
    ])

    # Start polling for updates
    await dp.start_polling(bot)

# Entry point
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
