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
import aiogram.utils.formatting as formatting

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

try:
    os.rmdir('db')
except:
    pass

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



@dp.message(F.document)
async def handle_pdf_file(message: types.Message):
    # Check if the file is a PDF
    if message.document.mime_type == ('application/pdf' or "application/txt"):
        
        msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNCVnQjEYjAwGtQeYOD5qd97Qn8FiwgACkAIAAsbveUXqtGM6TksDCjYE")
        
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
        
        # Step 2: Initialize Vector Database and Context
        global CONTEXT, CHUNKS, MODE, VECTOR_STORE
        add_to_vector_db(vector_db=VECTOR_STORE, docs=docs)
        CONTEXT = []
        CHUNKS = set()
        MODE = "chat_with_docs"
        
        await bot.delete_message(msg.chat.id, msg.message_id)
        await message.answer("Okay, let's talk about this file.")
        
    else:
        await message.answer("Please upload a PDF file.")




# HANDLER FOR INTERNET SEARCH




@dp.message(Command("search"))
async def handle_search(message: types.Message):
    
    global CONTEXT, CHUNKS, MODE
    CONTEXT = []
    CHUNKS = set()
    MODE = "internet_search"
    
    await message.answer("This is a demo of *HAOSearch \\(Hunt All Over Search\\)*\\. So ask me anything and will look for that in entire internet", parse_mode="MarkdownV2")
    



# MESSAGE HANDLER 




@flags.chat_action("typing")
@dp.message(F.text & ~F.text.startswith('/'))
async def send_message(message: types.Message):
    global CONTEXT, CHUNKS, CHUNKS_NUMBERS, MODE, VECTOR_STORE
    if MODE == "chat_with_docs":
        if len(CONTEXT) < 2:

            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")
            # store the unique chunks
            CHUNKS = list({doc.page_content: doc for doc in  retrieve_context(VECTOR_STORE, message.text, k=4)}.values())      
        
            full_context = "\n".join([doc.page_content for doc in CHUNKS])
            prompt = f"With help of this CONTEXT answer my question, try to answer structurally and covering all aspects of the question. \n\nCONTEXT:\n{full_context}\n\nQUESTION:\n{message.text}"
            
            response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=message.text, history=CONTEXT, prompt=prompt)
            
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
            
            response = transform_text(response)
            await bot.delete_message(msg.chat.id, msg.message_id)
            await message.answer(f"{response}", parse_mode="MarkdownV2")
            
        elif len(CONTEXT) >= 2:
            
            msg = await message.answer_sticker("CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")
            context = list({doc.page_content: doc for doc in  retrieve_context(VECTOR_STORE, message.text, k=4)}.values())
            
            # Create a set of contents from the first list for faster comparison
            existing_chunks = set(doc.page_content for doc in CHUNKS)

            # Filter out chunks from list2 that already exist in list1
            context = [doc for doc in context if doc.page_content not in existing_chunks]
        
            full_context = "\n".join([doc.page_content for doc in context])
            if context == []:
                prompt = f"{message.text}"
            else: 
                prompt = f"I have the following context in addition, can you help a bit more?\n\nCONTEXT:\n{full_context}\n\n\QUESTION:\n{message.text}"
            
            response, role, query, chat = generate_conversation_with_context(model = model, context=context, query=message.text, history=CONTEXT, prompt=prompt)
            
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
            
            response = transform_text(response)
            
            await bot.delete_message(msg.chat.id, msg.message_id)
            await message.answer(f"{response}", parse_mode="MarkdownV2")
    
    elif MODE == "internet_search":
        
        if len(CONTEXT) < 2:

            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNBdnQh--e8ycq2zhVm-A5zMS4eumbwACRgMAAiqHGURoXzCXdu7QsTYE")
            docs = search_info_to_docs(model, question=f"{message.text}")
            await bot.delete_message(msg.chat.id, msg.message_id)

            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNCVnQjEYjAwGtQeYOD5qd97Qn8FiwgACkAIAAsbveUXqtGM6TksDCjYE")            
            add_to_vector_db(VECTOR_STORE, docs)
            CHUNKS = retrieve_context(VECTOR_STORE, message.text, k=4)
            await bot.delete_message(msg.chat.id, msg.message_id)

            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENLYZnPbu98m8InnQC4bXdrzcTronVogACeQMAAo13GUSF6cj_mt_hyjYE")
            full_context = "\n\n\n".join([doc.page_content for doc in CHUNKS])
            prompt = f"Keeping provided CONTEXT in mind answer my question, your answer should be well structured and answer all aspects of the my question \n\nCONTEXT:\n{full_context}\n\n\QUESTION:\n{message.text}"
            
            response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=message.text, history=CONTEXT, prompt=prompt)
            
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
            
            # prepare the list of links
            links = []
            for doc in CHUNKS:
                try:
                    title = doc.metadata['title'] 
                except:
                    title = "Unknown URL"
                
                source = doc.metadata['source']
                
                links.append((source, title))
                
            links = list(set(links))
            links = '\n'.join(f'• [{transform_text(link[1])}]({link[0]})' for link in links)
            response = transform_text(response)
            text = f"{response}\n*Resources:*\n{links}"
            # for debug
            with open("text.txt", "w", encoding="utf-8") as file:
                file.write(text)
                
            await bot.delete_message(msg.chat.id, msg.message_id)
            await message.answer(text = text, parse_mode="MarkdownV2")
            print(CONTEXT)
        
        elif len(CONTEXT) >= 2: 
            
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")
            prompt = f"{message.text}"
            
            response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=message.text, history=CONTEXT, prompt=prompt)
            
            CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
            CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
            
            response = transform_text(response)
            
            await bot.delete_message(msg.chat.id, msg.message_id)
            await message.answer(f"{response}", parse_mode="MarkdownV2")
            print(CONTEXT)





@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    global VECTOR_STORE, CONTEXT, CHUNKS
    recreate_directory("db")
    VECTOR_STORE = Chroma(
        collection_name="collection",
        embedding_function=embeddings,
        persist_directory="db",  # Where to save data locally, remove if not necessary
    )
    CONTEXT = []
    CHUNKS = set()
    
    en_intro = """*Welcome to InfoSphere Telegram Bot*\\. To start our communication you can send me PDF or TXT file that you want to talk about\\.\nOr here is the functionality of this Bot that you can use:\n/start \\- Start the Bot again\\.\n/language \\- Change the language / Сменить язык\\.\n/search \\- HAOSearch \\- looks for the answer in entire internet\\."""
    ru_intro = """*Добро пожаловать в CookBook Telegram Бот*\\. Ниже написаны функции этого бота:\n/start \\- Запустить бота снова\\.\n/language \\- Change the language / Сменить язык\\.\n/search \\- HAOSearch \\- Искать ответ во всем интернете\\."""
    prompt = ru_intro if LNG == 'ru' else en_intro
    await message.answer(text=prompt, parse_mode="MarkdownV2")
    

async def main():
    
    await bot.set_my_commands([
        BotCommand(command="/search", description="HAOSearch any info in the internet"),
        BotCommand(command="/language", description="change the language"),
        BotCommand(command="/start", description="start again"),
    ])
    await dp.start_polling(bot)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())