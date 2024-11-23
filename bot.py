# imports

import os 
import logging
import aiogram
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import BotCommand, InlineKeyboardButton
from aiogram import Bot, Dispatcher, types, flags
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

MODE = None
CONTEXT = []
LNG = None

CHUNKS = set()
CHUNKS_NUMBERS = []
# LANGUAGE SETTINGS 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTOR_STORE = Chroma(
    collection_name="collection",
    embedding_function=embeddings,
    persist_directory="./db",  # Where to save data locally, remove if not necessary
)


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


@flags.chat_action("typing")
@dp.message(F.document)
async def handle_pdf_file(message: types.Message):
    # Check if the file is a PDF
    if message.document.mime_type == ('application/pdf' or "application/txt"):
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
        # for doc in docs:
        #     print(doc)
        #     print("-------")
        
        # Step 2: Initialize Vector Database and Context
        global CONTEXT, CHUNKS, MODE, VECTOR_STORE
        add_to_vector_db(vector_db=VECTOR_STORE, docs=docs)
        CONTEXT = []
        CHUNKS = set()
        MODE = "chat_with_docs"
        
        await message.answer("Okay, let's talk about this file.")
        
    else:
        await message.answer("Please upload a PDF file.")




# HANDLER FOR INTERNET SEARCH



@dp.message(Command("search"))
async def handle_search(message: types.Message):
    
    global CONTEXT, CHUNKS, MODE
    
    MODE = "internet_search"
    
    await message.answer("Ask me anything...")
    

# MESSAGE HANDLER 


@flags.chat_action("typing")
@dp.message(F.text & ~F.text.startswith('/'))
async def send_message(message: types.Message):
    global CONTEXT, CHUNKS, CHUNKS_NUMBERS, MODE, VECTOR_STORE
    if MODE == "chat_with_docs":
        if len(CONTEXT) < 2:
        
            # store the unique chunks
            CHUNKS = list({doc.page_content: doc for doc in  retrieve_context(VECTOR_STORE, message.text, k=4)}.values())

            # print("RETRIEVED CONTENT:")
            # for chunk in CHUNKS:
            #     print(chunk)  
            #     print("------")          
        
            full_context = "\n".join([doc.page_content for doc in CHUNKS])
            prompt = f"Relying on this Context answer my question. But keep in mind that you have to answer this question only relying on context i did provide:\n{full_context}\n\nQuestion:\n{message.text}"
            
            response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=message.text, history=CONTEXT, prompt=prompt)
            
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
            
            await message.answer(f"{response}")
            
        elif len(CONTEXT) >= 2:
            
            context = list({doc.page_content: doc for doc in  retrieve_context(VECTOR_STORE, message.text, k=4)}.values())
            
            # Create a set of contents from the first list for faster comparison
            existing_chunks = set(doc.page_content for doc in CHUNKS)

            # Filter out chunks from list2 that already exist in list1
            context = [doc for doc in context if doc.page_content not in existing_chunks]
            
            # print("\n\nCONTEXT RETRIEVED:")
            # for chunk in context:
            #     print(chunk)
            #     print("-------")
                
        
            full_context = "\n".join([doc.page_content for doc in context])
            if context == []:
                prompt = f"{message.text}"
            else: 
                prompt = f"I have the following context in addition, can you help a bit more?\nContext:\n{full_context}\n\nQuestion:\n{message.text}"
            
            response, role, query, chat = generate_conversation_with_context(model = model, context=context, query=message.text, history=CONTEXT, prompt=prompt)
            
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
            
            await message.answer(f"{response}")
    
    elif MODE == "internet_search":
        
        if len(CONTEXT) < 2:
            # perform search
            docs = search_info_to_docs(model, question=f"{message.text}")
            
            add_to_vector_db(VECTOR_STORE, docs)
            
            CHUNKS = retrieve_context(VECTOR_STORE, message.text, k=4)

            full_context = "\n\n\n".join([doc.page_content for doc in CHUNKS])
            prompt = f"Relying on this Context answer my question. \nContext:\n{full_context}\n\nQuestion:\n{message.text}"
            
            response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=message.text, history=CONTEXT, prompt=prompt)
            
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
            
            await message.answer(f"{response}\n\nResources: {[doc.metadata['source'] for doc in CHUNKS]}")
        
        elif len(CONTEXT) >= 2: 
            
            # context = retrieve_context(VECTOR_STORE, message.text, k=4)
                
            # Create a set of contents from the first list for faster comparison
            # existing_chunks = set(doc.page_content for doc in CHUNKS)

            # Filter out chunks from list2 that already exist in list1
            # context = [doc for doc in context if doc.page_content not in existing_chunks]
            
            # print("\n\nCONTEXT RETRIEVED:")
            # for chunk in context:
            #     print(chunk)
            #     print("-------")
                
            prompt = f"{message.text}"
            
            response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=message.text, history=CONTEXT, prompt=prompt)
            
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
            
            await message.answer(f"{response}")

# Example of existing commands for context



@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    en_intro = """*Welcome to InfoSphere Telegram Bot*\\. Here is the functionality of this Bot:\n/start \\- Start the Bot again\\.\n/language \\- Change the language / Сменить язык\\.\n/search \\- Search for the answer in entire internet\\."""
    ru_intro = """*Добро пожаловать в CookBook Telegram Бот*\\. Ниже написаны функции это бота:\n/start \\- Запустить бота снова\\.\n/language \\- Change the language / Сменить язык\\.\n/search \\- Искать ответ во всем интернете\\."""
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