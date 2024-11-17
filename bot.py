# imports

import os 
import logging
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import BotCommand, InlineKeyboardButton
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from envvar import BOT_TOKEN
from functions.functions import *
from aiogram import F
from config import model

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

CONTEXT = []
LNG = None

vector_db = None
CHUNKS = []
CHUNKS_NUMBERS = []
# LANGUAGE SETTINGS 



@dp.message(Command("language"))
async def set_language(message: types.Message):    
    buttons = [
        [
            types.InlineKeyboardButton(text="English", callback_data="en"),
            types.InlineKeyboardButton(text="Русский", callback_data="ru")
        ],
    ]
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    
    await message.answer(
        "Choose the language / Выберите язык:",
        reply_markup=keyboard
    )

@dp.callback_query(F.data == "en")
async def lang_change(callback: types.CallbackQuery):
    global LNG
    LNG = 'en'
    await callback.answer()  # Acknowledging the callback
    await callback.message.answer("Language changed to English")
    
@dp.callback_query(F.data == "ru")
async def lang_change(callback: types.CallbackQuery):
    global LNG
    LNG = 'ru'
    await callback.answer()  # Acknowledging the callback
    await callback.message.answer("Язык изменен на Русский")



# HANDLER FOR RECEIVING PDF FILES



@dp.message(F.document)
async def handle_pdf_file(message: types.Message):
        # Check if the file is a PDF
    if message.document.mime_type == 'application/pdf':
        # Create the 'documents' folder if it doesn't exist
        os.makedirs("documents", exist_ok=True)
        
        # Get the file information
        file_id = message.document.file_id
        file_name = message.document.file_name
        file_path = f"documents/{file_name}"
        
        # Download and save the file
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, file_path)
        
        # Step 1: Upload Document
        docs = upload_document(file_path)
        for doc in docs:
            print(doc)
            print("-------")
        
        # Step 2: Initialize Vector Database and Context
        vector_db_ = initialize_vector_db(docs)
        global vector_db, CONTEXT, CHUNKS
        vector_db = vector_db_
        CONTEXT = []
        CHUNKS = []
        
        await message.answer("Okey, let's talk about this PDF file.")
        
    else:
        await message.answer("Please upload a PDF file.")




# MESSAGE HANDLER 




@dp.message(F.text & ~F.text.startswith('/'))
async def send_message(message: types.Message):
    global CONTEXT, CHUNKS, CHUNKS_NUMBERS
    if len(CONTEXT) < 2:
    
        CHUNKS = retrieve_context(vector_db, message.text, k=4)
        
        print("\n\nCONTEXT RETRIEVED:")
        for part in CHUNKS:
            CHUNKS_NUMBERS.append(part.metadata['page'])
            print(part)
            print("-------------")
        
        full_context = "\n".join([doc.page_content for doc in CHUNKS])
        prompt = f"Relying on this Context answer my question. But keep in mind that you have to answer this question only relying on context i did provide:\n{full_context}\n\nQuestion:\n{message.text}"
        
        response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=message.text, history=CONTEXT, prompt=prompt)
        
        CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
        CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
        
        await message.answer(f"{response}")
        
    elif len(CONTEXT) >= 2:
        
        context = retrieve_context(vector_db, message.text, k=4)
        
        print("\n\nCONTEXT RETRIEVED:")
        for part in context:
            if part.metadata['page'] in CHUNKS_NUMBERS:
                index = context.index(part.metadata['page'])
                context.remove(part.metadata['page'])
                del context[index]
            print(part)
            print("-------------")
        
        full_context = "\n".join([doc.page_content for doc in context])
        prompt = f"I have the following context in addition, can you help a bit more?\nContext:\n{full_context}\n\nQuestion:\n{message.text}"
        
        response, role, query, chat = generate_conversation_with_context(model = model, context=context, query=message.text, history=CONTEXT, prompt=prompt)
        
        CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
        CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
        
        await message.answer(f"{response}")
    


# Example of existing commands for context



@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    en_intro = """*Welcome to InfoSphere Telegram Bot* \n\nHere is the list of commands\\.\\.\\."""
    ru_intro = """*Добро пожаловать в CookBook Telegram Бот* \n\nНиже расположены команды\\.\\.\\."""
    prompt = ru_intro if LNG == 'ru' else en_intro
    await message.answer(text=prompt, parse_mode="MarkdownV2")

async def main():
    
    await bot.set_my_commands([
        BotCommand(command="/search", description="search any info in the internet"),
        BotCommand(command="/language", description="change the language"),
        BotCommand(command="/start", description="start again"),
    ])
    await dp.start_polling(bot)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())