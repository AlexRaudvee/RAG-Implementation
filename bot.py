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
from langchain_community.vectorstores import DocArrayInMemorySearch

# Configure logging
logging.basicConfig(level=logging.INFO)  # Logs important information for debugging and monitoring

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)  # Initialize the bot with the provided token
dp = Dispatcher()  # Dispatcher to manage and handle events and updates

# Global Variables
MODE = None  # Mode of operation: 'chat_with_docs' or 'internet_search'
CONTEXT = []  # Conversation history for contextual replies
LNG = "ru"  # Current language setting ('en' or 'ru')
CHUNKS = set()  # Unique document chunks used for context
CHUNKS_NUMBERS = []  # Track chunk numbers for efficient management

# Initialize HuggingFace embeddings


# Initialize vector database
try:
    os.rmdir('db')  # Remove the database directory if it exists
except:
    pass


VECTOR_STORE = None



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

# Handler for receiving files
@dp.message(F.document)
async def handle_files(message: types.Message):
    """
    Handle uploaded PDF or TXT files, process them, and initialize context for conversation.

    Args:
        message (types.Message): Telegram message containing the file.
    """
    # Check if the uploaded file is a PDF or TXT
    if message.document.mime_type in ('application/pdf', 'application/txt', 'text/plain'):
        # Send a sticker as feedback
        msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNCVnQjEYjAwGtQeYOD5qd97Qn8FiwgACkAIAAsbveUXqtGM6TksDCjYE")
    
        # Retrieve file information
        file_id = message.document.file_id

        # Download and save the file locally
        file = await bot.get_file(file_id)

        # Process the document
        docs = upload_document(file.file_path)  # Custom function to upload and parse the document
        
        # Update global variables
        global CONTEXT, CHUNKS, MODE, VECTOR_STORE
        VECTOR_STORE = add_to_vector_db(vector_db=VECTOR_STORE, docs=docs)  # Add document to the vector database
        CONTEXT = []
        CHUNKS = set()
        MODE = "chat_with_docs"  # Set mode to document-based chat

        # Delete the sticker and send confirmation
        await bot.delete_message(msg.chat.id, msg.message_id)
        
        if LNG == "en": 
            await message.answer("Okay, let's talk about this file.")
        elif LNG == "ru":
            await message.answer("Хорошо, давайте поговорим о этом файле.")
    else:
        # Notify the user if the file is not a PDF or TXT
        if LNG == "en":
            await message.answer("Please upload a valid PDF or TXT file.")
        elif LNG == "ru":
            await message.answer("Пожалуйста, загрузите валидный PDF или TXT файл")
            
# Handler for internet search
@dp.message(Command("search"))
async def handle_search(message: types.Message):
    """
    Handle the /search command to activate the internet search mode.

    Args:
        message (types.Message): Telegram message object.
    """
    global CONTEXT, CHUNKS, MODE, VECTOR_STORE
    CONTEXT = []  # Reset context
    CHUNKS = set()  # Reset chunks
    MODE = "internet_search"  # Set mode to internet search
    VECTOR_STORE = None # Reset the Vec DB

    # Inform the user about the activated mode
    if LNG == "en":
        await message.answer("This is a demo of *HAOSearch \\(Hunt All Over Search\\)*\\. So ask me anything and I will look for that in the entire internet\\.", parse_mode="MarkdownV2")
    elif LNG == "ru":
        await message.answer("Это демонстрация *HAOSearch \\(Hunt All Over Search\\)*\\. Так что спрашивайте меня о чем угодно, и я поищу это во всем Интернете\\.", parse_mode="MarkdownV2")

# General message handler
@dp.message(F.text & ~F.text.startswith('/'))
async def send_message(message: types.Message):
    """
    Handle user messages and generate a response based on the current mode (document chat or internet search).

    Args:
        message (types.Message): Telegram message containing text input.
    """
    global CONTEXT, CHUNKS, CHUNKS_NUMBERS, MODE, VECTOR_STORE
    links_in_text = False
    
    if contains_links(message.text):
        MODE = "chat_with_links"
        links_in_text = True
    
    if MODE == "chat_with_docs":
        # Respond with document-based context
        if len(CONTEXT) < 2:
            # Show typing feedback and retrieve context
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")
            CHUNKS = list({doc.page_content: doc for doc in retrieve_context(VECTOR_STORE, message.text, k=4)}.values())
            await bot.delete_message(msg.chat.id, msg.message_id)
            
            # Create a prompt and generate response
            full_context = "\n".join([doc.page_content for doc in CHUNKS])
            prompt = f"With the provided CONTEXT, answer the following QUESTION, your answer should well structured in bullet points and answer all aspects of the question:\n\nCONTEXT:\n{full_context}\n\nQUESTION:\n{message.text}"
            
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENLYZnPbu98m8InnQC4bXdrzcTronVogACeQMAAo13GUSF6cj_mt_hyjYE")
            response, _, _, _ = generate_conversation_with_context(model, CHUNKS, message.text, CONTEXT, prompt)

            # Save the conversation context
            CONTEXT.extend([
                {'role': 'user', 'parts': [prompt]},
                {'role': 'model', 'parts': [response]}
            ])
            
            if LNG == 'ru':
                response = translate(context=response, to_lang = 'ru')
            
            # Send the response to the user
            response = transform_text(response)
            
            # for debug purposes
            with open("text.txt", mode='w') as file:
                file.write(response)
                
            await bot.delete_message(msg.chat.id, msg.message_id)
            # answer to the user
            if len(response) > 4095:
                for x in range(0, len(response), 4095):
                    await message.answer(text=response[x:x+4095], parse_mode="MarkdownV2")
            else:
                await message.answer(text=response, parse_mode="MarkdownV2")
                
                     
        elif len(CONTEXT) >= 2:
            # Show typing feedback and retrieve context
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")
            CHUNKS = list({doc.page_content: doc for doc in retrieve_context(VECTOR_STORE, message.text, k=4)}.values())
            await bot.delete_message(msg.chat.id, msg.message_id)
            
            # Create a prompt and generate response
            full_context = "\n".join([doc.page_content for doc in CHUNKS])
            prompt = f"I have the following CONTEXT in addition, answer the following QUESTION, your answer should well structured in bullet points and answer all aspects of the question:\n\nCONTEXT:\n{full_context}\n\nQUESTION:\n{message.text}"
            
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENLYZnPbu98m8InnQC4bXdrzcTronVogACeQMAAo13GUSF6cj_mt_hyjYE")
            response, _, _, _ = generate_conversation_with_context(model, CHUNKS, message.text, CONTEXT, prompt)

            # Save the conversation context
            CONTEXT.extend([
                {'role': 'user', 'parts': [prompt]},
                {'role': 'model', 'parts': [response]}
            ])
            
            if LNG == 'ru':
                response = translate(context=response, to_lang = 'ru')
            
            # Send the response to the user
            response = transform_text(response)
            await bot.delete_message(msg.chat.id, msg.message_id)
            # answer to the user
            if len(response) > 4095:
                for x in range(0, len(response), 4095):
                    await message.answer(text=response[x:x+4095], parse_mode="MarkdownV2")
            else:
                await message.answer(text=response, parse_mode="MarkdownV2")

    elif MODE == "internet_search":

        # Ensure the conversation history context is minimal for simplicity
        if len(CONTEXT) < 2:

            # Step 1: Notify the user about the search process by sending a sticker
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNBdnQh--e8ycq2zhVm-A5zMS4eumbwACRgMAAiqHGURoXzCXdu7QsTYE")

            # Use a custom function to perform the internet search and convert results to documents
            docs = search_info_to_docs(model, question=f"{translate(message.text, to_lang='en')}", lng="en")

            # Remove the initial sticker message once the task is done
            await bot.delete_message(msg.chat.id, msg.message_id)

            # Step 2: Notify the user about embedding the search results
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNCVnQjEYjAwGtQeYOD5qd97Qn8FiwgACkAIAAsbveUXqtGM6TksDCjYE")

            # Add the retrieved documents to the vector database for embedding and context retrieval
            VECTOR_STORE = add_to_vector_db(VECTOR_STORE, docs)

            # Retrieve the most relevant chunks of information from the vector database
            CHUNKS = retrieve_context(VECTOR_STORE, translate(message.text, to_lang='en'), k=4)

            # Remove the second sticker after embedding is complete
            await bot.delete_message(msg.chat.id, msg.message_id)

            # Step 3: Notify the user that the response is being generated
            msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENLYZnPbu98m8InnQC4bXdrzcTronVogACeQMAAo13GUSF6cj_mt_hyjYE")

            # Combine the retrieved chunks to form a single context string
            full_context = "\n\n\n".join([doc.page_content for doc in CHUNKS])

            # Create a prompt for the model using the context and user question
            prompt = f"""Keeping provided CONTEXT in mind answer my question, your answer should be well structured with bullet points and answer all aspects of my question 
                        \n\nCONTEXT:\n{full_context}\n\nQUESTION:\n{translate(message.text, to_lang='en')}"""

            # Generate a response using a custom function with the model, context, and prompt
            response, role, query, chat = generate_conversation_with_context(
                model=model, 
                context=CHUNKS, 
                query=translate(message.text, 'en'), 
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
                links.append((source.replace("(", "\\(").replace(")", "\\)"), title))  # Add to the list of links
            
            # Remove duplicate links and format them as a markdown list
            links = list(set(links))
            links = '\n'.join(f'• [{transform_text(link[1])}]({link[0]})' for link in links)

            if LNG == 'ru':
                response = translate(context=response, to_lang='ru')
                
            # Transform the response text for better formatting
            response = transform_text(response)

            # Combine the response and resources into the final message
            if LNG == "en":
                text = f"{response}\n*Resources:*\n{links}"
            elif LNG == 'ru':
                text = f"{response}\n*Источники:*\n{links}"
                
            # for debug purposes
            with open("text.txt", mode='w') as file:
                file.write(text)
                
            # Remove the sticker used during response generation
            await bot.delete_message(msg.chat.id, msg.message_id)

            # Send the final formatted response to the user
            if len(text) > 4095:
                for x in range(0, len(text), 4095):
                    await message.answer(text=text[x:x+4095], parse_mode="MarkdownV2")
            else:
                await message.answer(text=text, parse_mode="MarkdownV2")

        elif len(CONTEXT) >= 2:
            
            match = re.search(r"QUESTION:\s*(.*)", CONTEXT[-2:][0]['parts'][0], re.DOTALL)
            question = match.group(1).strip()            
            search_again = bool(float(model.generate_content(contents=f"Given my last two questions: '{question}' and '{message.text}'. return me 0 if these questions are related to the same topic and 1 otherwise, your answer should be only 0 or 1 no words or letters", generation_config={"temperature": 1}).candidates[0].content.parts[0].text))
            
            if search_again:
                # Step 1: Notify the user about the search process by sending a sticker
                msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNBdnQh--e8ycq2zhVm-A5zMS4eumbwACRgMAAiqHGURoXzCXdu7QsTYE")

                # Use a custom function to perform the internet search and convert results to documents
                docs = search_info_to_docs(model, question=f"{translate(message.text, to_lang='en')}", lng=LNG)

                # Remove the initial sticker message once the task is done
                await bot.delete_message(msg.chat.id, msg.message_id)

                # Step 2: Notify the user about embedding the search results
                msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNCVnQjEYjAwGtQeYOD5qd97Qn8FiwgACkAIAAsbveUXqtGM6TksDCjYE")

                # Add the retrieved documents to the vector database for embedding and context retrieval
                VECTOR_STORE = add_to_vector_db(VECTOR_STORE, docs)

                # Retrieve the most relevant chunks of information from the vector database
                CHUNKS = retrieve_context(VECTOR_STORE, translate(message.text, to_lang='en'), k=2)

                # Remove the second sticker after embedding is complete
                await bot.delete_message(msg.chat.id, msg.message_id)

                # Step 3: Notify the user that the response is being generated
                msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENLYZnPbu98m8InnQC4bXdrzcTronVogACeQMAAo13GUSF6cj_mt_hyjYE")

                # Combine the retrieved chunks to form a single context string
                full_context = "\n\n\n".join([doc.page_content for doc in CHUNKS])

                # Create a prompt for the model using the context and user question
                prompt = f"""Keeping provided CONTEXT in mind answer my question, your answer should be well structured with bullet points and answer all aspects of my question 
                            \n\nCONTEXT:\n{full_context}\n\nQUESTION:\n{translate(message.text, to_lang='en')}"""

                # Generate a response using a custom function with the model, context, and prompt
                response, role, query, chat = generate_conversation_with_context(
                    model=model, 
                    context=CHUNKS, 
                    query=translate(message.text, to_lang='en'), 
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
                    links.append((source.replace("(", "\\(").replace(")", "\\)"), title))  # Add to the list of links
                
                # Remove duplicate links and format them as a markdown list
                links = list(set(links))
                links = '\n'.join(f'• [{transform_text(link[1])}]({link[0]})' for link in links)

                if LNG == "ru":
                    response = translate(context=response, to_lang='ru')
                
                # Transform the response text for better formatting
                response = transform_text(response)

                # Combine the response and resources into the final message
                if LNG == "en":
                    text = f"{response}\n*Resources:*\n{links}"
                elif LNG == "ru":
                    text = f"{response}\n*Источники:*\n{links}"
                    
                # for debug purposes
                with open("text.txt", mode='w') as file:
                    file.write(text)
                    
                # Remove the sticker used during response generation
                await bot.delete_message(msg.chat.id, msg.message_id)

                # Send the final formatted response to the user
                # Send the final formatted response to the user
                if len(text) > 4095:
                    for x in range(0, len(text), 4095):
                        await message.answer(text=text[x:x+4095], parse_mode="MarkdownV2")
                else:
                    await message.answer(text=text, parse_mode="MarkdownV2")
                
            else:
                # notify user about writing of the message by sending a sticker
                msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")

                prompt = f"QUESTION:{translate(message.text, to_lang='en')}"
                
                # generate response
                response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=translate(message.text, to_lang='en'), history=CONTEXT, prompt=prompt)
            
                # add everything context
                CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
                CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
                
                if LNG == "ru":
                    response = translate(context=response, to_lang='ru')
                    
                # transform text from plain to markdown
                response = transform_text(response)
                
                # delete the sticker
                await bot.delete_message(msg.chat.id, msg.message_id)
                # answer to the user
                if len(response) > 4095:
                    for x in range(0, len(response), 4095):
                        await message.answer(text=response[x:x+4095], parse_mode="MarkdownV2")
                else:
                    await message.answer(text=response, parse_mode="MarkdownV2")
    
    elif MODE == "chat_with_links":
        
        
        # Show typing feedback and retrieve context
        msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")
        
        did_not_read_url = []
        docs = []
        urls, question = extract_urls_and_clean_text(message.text)
        for url in urls:
            loader = WebBaseLoader(f"{url}", bs_get_text_kwargs={"strip": True})
        
            try: 
                doc = loader.load()            
                docs.append(doc[0])
            except:
                print(f"failed to read: {url}")
                did_not_read_url.append(url)
                continue
            
        if links_in_text:
            prompt = f"""
                Given the following context:
                
                CONTEXT:{docs[0].page_content}
                
                Answer my question: 
                
                QUESTION:{question}
                
                Your answer should be concise and well structured such that it covers all aspects of the question.
            """
        else: 
            prompt = translate(message.text, to_lang='en')
            
        # generate response
        response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=translate(message.text, to_lang='en'), history=CONTEXT, prompt=prompt)

        # add everything context
        CONTEXT.append({'role': 'user', 'parts': [f'{prompt}']})
        CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
        
        if LNG == "ru":
            response = translate(context=response, to_lang='ru')
            
        # transform text from plain to markdown
        response = transform_text(response)
        
        # delete the sticker
        await bot.delete_message(msg.chat.id, msg.message_id)
        # answer to the user
        if len(response) > 4095:
            for x in range(0, len(response), 4095):
                await message.answer(text=response[x:x+4095], parse_mode="MarkdownV2")
        else:
            await message.answer(text=response, parse_mode="MarkdownV2")
                        
    elif MODE == None: 
        
        msg = await message.answer_sticker(sticker="CAACAgEAAxkBAAENNQ9nQzyhLrpAveOj8j6J4hWI7jUngQACIgMAAma-oUY566OY856vSzYE")

        response, role, query, chat = generate_conversation_with_context(model = model, context=CHUNKS, query=translate(message.text, to_lang='en'), history=CONTEXT, prompt=translate(message.text, to_lang='en'))
        
        # add everything context
        CONTEXT.append({'role': 'user', 'parts': [f"{translate(message.text, to_lang='en')}"]})
        CONTEXT.append({'role': 'model', 'parts': [f"{response}"]})
        
        if LNG == "ru":
            response = translate(context=response, to_lang='ru')
            
        # transform text from plain to markdown
        response = transform_text(response)
        
        # delete the sticker
        await bot.delete_message(msg.chat.id, msg.message_id)
        
        # answer to the user
        if len(response) > 4095:
            for x in range(0, len(response), 4095):
                await message.answer(text=response[x:x+4095], parse_mode="MarkdownV2")
        else:
            await message.answer(text=response, parse_mode="MarkdownV2")
                

# Command: /start
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    """
    Send a welcome message and reset the bot to its initial state.

    Args:
        message (types.Message): Telegram message triggering the command.
    """
    global VECTOR_STORE, CONTEXT, CHUNKS, MODE
    
    # Clear the database directory
    try:
        shutil.rmtree("./db")
    except: 
        pass
    
    VECTOR_STORE = None
    CONTEXT = []
    CHUNKS = set()
    MODE = None

    # Select the introduction based on the language
    en_intro = ("*Welcome to InfoSphere Telegram Bot*\\. To start, you can send a PDF or TXT file, "
                "or use the following commands:\n/start \\- Restart the bot\\.\n/language \\- Change the language\\.\n/search \\- HAOSearch the internet\\.\n/help \\- this command is going to display some help notes\\.")
    ru_intro = ("*Добро пожаловать в InfoSphere Телеграм Бот*\\. Вы можете загрузить PDF или TXT файл или использовать команды:"
                "\n/start \\- Запустить бота\\.\n/language \\- Сменить язык\\.\n/search \\- Искать ответ в интернете\\.\n/help \\- эта команда покажет небольшую инструкцию о том как использовать бота\\.")
    prompt = ru_intro if LNG == 'ru' else en_intro

    await message.answer(text=prompt, parse_mode="MarkdownV2")

# Command: /help 
@dp.message(Command("help"))
async def send_help(message: types.Message):
    """
    Send a help message
    """
    help_text_en = "*Help*\\.\n • This Bot is aimed to help you with search of any information, you can upload your own documents \\(PDF, TXT\\), and answer questions about this documents, the bot is going to give you answer relying on the context from this documents\\.\n • In addition you can use command /search, and then type your question, then the bot is going to look for information in entire Internet \\(HAOSearch\\)\\. \nYou can use the following commands:\n/start \\- This command to restart the bot\\.\n/search \\- This command activates the HAOSearch mode, when the bot is able to look out online resources to answer your question\\.\n/help \\- Holp command which returns all the commands and instruction for this bot\\."
    help_text_ru = "*Помощь*\\.\n • Этот бот призван помочь вам в поиске любой информации, вы можете загружать свои собственные документы \\(PDF, TXT\\), и отвечать на вопросы по этим документам, бот собирается чтобы дать вам ответ, основываясь на контексте из этого документа\\.\n • Кроме того, вы можете использовать команду /search, а затем ввести свой вопрос, тогда бот будет искать информацию во всем Интернете \\(HAOSearch\\)\\. \nВы можете использовать следующее команды:\n/start \\- Эта команда для перезапуска бота\\.\n/search \\- Эта команда активирует режим HAOSearch, когда бот может искать онлайн-ресурсы, чтобы ответить на ваш вопрос\\.\ n/help \\- команда Holp, которая возвращает все команды и инструкции для этого бота\\."
    
    if LNG == 'en':
        await message.answer(help_text_en, parse_mode="MarkdownV2")
    elif LNG == 'ru':
        await message.answer(help_text_ru, parse_mode="MarkdownV2")
    
# Main function
async def main():
    """
    Main entry point for starting the bot.
    """
    # Set bot commands
    await bot.set_my_commands([
        BotCommand(command="/search", description="HAOSearch in internet"),
        BotCommand(command="/language", description="Change language"),
        BotCommand(command="/start", description="Restart the InfoSearch"),
        BotCommand(command="/help", description="How to use the InfoSphere"),
    ])

    # Start polling for updates
    await dp.start_polling(bot)

# Entry point
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
