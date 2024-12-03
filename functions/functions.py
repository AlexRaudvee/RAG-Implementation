import io
import os
import re
import torch
import shutil
import PyPDF2
import requests
import urllib.request
import translators as ts

from uuid import uuid4
from bs4 import BeautifulSoup
from tokenizers import Tokenizer
from googlesearch import search 
from semantic_text_splitter import TextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders.telegram import text_to_docs
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

from config import BOT_TOKEN, embeddings


def upload_document(file_path):
    print("CHUNKING OF DOCUMENT: ...")
    # URL of the file
    URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    
    # Check file type
    if URL.endswith(".txt"):
        # Send a GET request
        response = requests.get(URL)

        # Ensure the request was successful
        if response.status_code == 200:
            text = response.text
            
    elif URL.endswith(".pdf"):
        # Fetch the PDF file as bytes
        req = urllib.request.Request(URL, headers={'User-Agent': "Mozilla/5.0"})
        remote_file = urllib.request.urlopen(req).read()

        # Convert bytes to a file-like object
        remote_file_bytes = io.BytesIO(remote_file)
        # Read and process the PDF
        pdf_reader = PyPDF2.PdfReader(remote_file_bytes)
        text = ""

        # Extract text from each page
        for page in pdf_reader.pages:
            text += page.extract_text()
        
    else:
        raise NotImplementedError("Unsupported file type. Only .txt and .pdf are supported.")
    
    # Split documents into chunks
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, capacity=(50, 1000), overlap=25)

    chunks = splitter.chunks(text=text)
    docs = text_to_docs(chunks)
            
    print("CHUNKING COMPLETE!")
    
    return docs


def add_to_vector_db(vector_db, docs):
    print("EMBEDDING THE DOCUMENT: ...")
    
    # uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    
    print("EMBEDDING DONE!")
        
    return vector_db


def retrieve_context(vector_db, query: str, k: int=4):
    return vector_db.similarity_search(query=query, k=k)


def generate_response_with_context(model, context, query):
    print("PULING KNOWLEDGE TO MODEL: ...")
    full_context = "\n".join([doc.page_content for doc in context])
    prompt = f"Relying on this Context answer my question. But keep in mind that you have to answer this question only relying on this context:\n{full_context}\n\nQuestion:\n{query}"
    
    try:
        _response = model.generate_content(prompt)
        response = _response.candidates[0].content.parts[0].text
        role = "model"
        print("PULL OF KNOWLEDGE DONE!")
        
        return response, role, query

    except Exception as e:
        print(f"Error encountered: {e}")
        return f"Failure: {e}", "model", query
    
    
def generate_conversation_with_context(model, context: list, query: str, history: list, prompt: str):
        
    chat = model.start_chat(
            history=history,
        )
    
    try:
        
        _response = chat.send_message(prompt)

        response = _response.candidates[0].content.parts[0].text
        role = "model"
        
        return response, role, query, chat

    except Exception as e:
        return f"Failure: {e}", "model", query, chat


def google_search(query: str):

    list_of = []

    for link in search(query=query, num=8, stop=8):
        list_of.append(link)

    for link in search(query=query, num=1, stop=1, tbs="qdr:d"):
        list_of.append(link)

    
    return list(set(list_of))


def parse_and_save(url, output_filename="output.txt"):
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract all visible text
        text = soup.get_text(strip=True)

        # Extract all links from the webpage
        links = []# [a['href'] for a in soup.find_all('a', href=True) if "https://" in a['href']]

        # Save text to a file
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(text)

        # Get the absolute file path
        file_path = os.path.abspath(output_filename)

        # Return the file path and the list of links
        return file_path, links, url
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, [], url
    
    
# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = similarity_model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# def get_relevant_links(question: str, links: list[str], similarity: float = 0.7):
#     # Get the embedding for the question
#     question_embedding = get_embedding(question)

#     # Filter out the relevant links based on similarity
#     relevant_links = []
#     for link in links:
#         metadata_embedding = get_embedding(link)
        
#         # Compute cosine similarity between question and metadata
#         similarity_ = cosine_similarity([question_embedding], [metadata_embedding])[0][0]
        
#         # Set a threshold for similarity
#         if similarity_ > similarity:  # You can adjust this threshold
#             relevant_links.append(link)
            
#     return relevant_links
    
    
def search_info_to_docs(model, question: str, lng: str = "en"):
        
    if lng == 'ru':
        question = translate(question, to_lang='en')
        
    prompt = f"i have the following question: {question}. Generate me a list of search queries that i can put in google. Your response should contain the python list only wit at most 2 strings"

    _response = model.generate_content(prompt, generation_config={"temperature": 0})
    questions = _response.candidates[0].content.parts[0].text
    
    # Use regex to extract all strings inside the quotes
    pattern = r'["\'](.*?)["\']'
    questions = re.findall(pattern, questions)
    
    links_visited = 0
    docs = []
    for question in questions:
        if links_visited >= 8:
            return docs
        
        links_visited_on_1 = 0
        for link in search(query=question, lang=lng, num=10, stop=10):
            
            loader = WebBaseLoader(f"{link}", bs_get_text_kwargs={"strip": True})
            
            try: 
                doc = loader.load()
                if doc not in docs:
                    links_visited += 1
                    links_visited_on_1 += 1
                    docs += doc
                    
                if links_visited_on_1 == 4:
                    break
            except:
                print(f"failed to read: {link}")
                continue
            
    return docs


def links_list_to_message(input_list):
    # Join each string with a prefix of "- " followed by a newline
    return "\n".join(f"• [{item[1]}]({item[0]})" for item in input_list)


def transform_text(text: str) -> str:    
    # Replace single asterisks (*) with '>'
    text = text.replace("\n*", "\n\n•")
    text = text.replace("\n    *", "\n    •")
    text = text.replace("\n   *", "\n   •")
    text = text.replace("\n  *", "\n  •")
    text = text.replace("   *", "   •")
    text = text.replace("  *", "  •")
    # Replace double asterisks (**) with single asterisks (*)
    text = text.replace("**", "*")
    # Place a backslash before specific characters
    characters_to_escape = ["(", ")", "[", "]", "{", "}", "-", ".", "!", "|", "+", "~"]
    for char in characters_to_escape:
        text = text.replace(char, f"\\{char}")
    
    return text


def recreate_directory(path):
    """
    Deletes the directory if it exists and creates a new, empty one.
    """
    if os.path.exists(path):
        # Remove the directory and its contents
        shutil.rmtree(path)
        print(f"Deleted existing directory: {path}")

    # Create a new empty directory
    os.makedirs(path)
    print(f"Created new directory: {path}")
    
    
def translate(context: str, html: bool = False, to_lang: str = "ru") -> str:

    # Perform the translation
    if html:
        translated_text = ts.translate_html(html_text=context, translator='yandex', to_language=to_lang)
    else:
        translated_text = ts.translate_text(query_text=context, translator='yandex', to_language=to_lang)
    
    return translated_text

def extract_urls_and_clean_text(text):
    # Regular expression for URLs and domain names
    pattern = r"(https?://[^\s]+|www\.[^\s]+|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})"
    
    # Find all matches (URLs and domains)
    urls = re.findall(pattern, text)
    
    # Remove URLs/domains from the text
    clean_text = re.sub(pattern, '', text).strip()
    
    return urls, clean_text

def contains_links(text):
    # Regular expression for detecting URLs or domain names
    pattern = r"(https?://[^\s]+|www\.[^\s]+|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})"
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # Return True if a match is found, otherwise False
    return bool(match)