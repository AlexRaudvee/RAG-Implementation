import os
import torch
import shutil
import requests

from uuid import uuid4
from bs4 import BeautifulSoup
from tokenizers import Tokenizer
from googlesearch import search 
from semantic_text_splitter import TextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders.telegram import text_to_docs
from langchain_community.document_loaders import WebBaseLoader

from config import tokenizer, similarity_model


def upload_document(file_path, source: str | None = None):
    print("CHUNKING OF DOCUMENT: ...")
    
    # Check file type
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path, concatenate_pages=True)
        documents = loader.load()
        text = documents[0].page_content
        
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
    
    uuids = [str(uuid4()) for _ in range(len(docs))]

    vector_db.add_documents(documents=docs, ids=uuids)  
    
    print("EMBEDDING DONE!")
        
    return vector_db


def retrieve_context(vector_db, query: str, k: int=3):
    return vector_db.similarity_search(query, k=k)


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

    for link in search(query=query, num=5, stop=5):
        list_of.append(link)
    for link in search(query=query, num=1, stop=1, tbs="qdr:h"):
        list_of.append(link)
    for link in search(query=query, num=1, stop=1, tbs="qdr:d"):
        list_of.append(link)
    for link in search(query=query, num=1, stop=1, tbs="qdr:m"):
        list_of.append(link)
        
    return list(set(list_of) )
        
    
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
    
    
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = similarity_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def get_relevant_links(question: str, links: list[str], similarity: float = 0.7):
    # Get the embedding for the question
    question_embedding = get_embedding(question)

    # Filter out the relevant links based on similarity
    relevant_links = []
    for link in links:
        metadata_embedding = get_embedding(link)
        
        # Compute cosine similarity between question and metadata
        similarity_ = cosine_similarity([question_embedding], [metadata_embedding])[0][0]
        
        # Set a threshold for similarity
        if similarity_ > similarity:  # You can adjust this threshold
            relevant_links.append(link)
            
    return relevant_links
    
    
def search_info_to_docs(model, question: str):
        
    prompt = f"i have this question:'{question}'. Rephrase it such that it suits better as a google search prompt. Your answer should contain only rephrased prompt!"

    _response = model.generate_content(prompt)
    question = _response.candidates[0].content.parts[0].text

    for link in google_search(question):
        loader = WebBaseLoader(f"{link}", bs_get_text_kwargs={"strip": True})
    
        try:
            docs = loader.load()
        except:
            print(f"FAILED TO READ: {link}")
            continue
        
    return docs


def links_list_to_message(input_list):
    # Join each string with a prefix of "- " followed by a newline
    return "\n".join(f"• [{item[1]}]({item[0]})" for item in input_list)


def transform_text(text: str) -> str:    
    # Replace single asterisks (*) with '>'
    text = text.replace("\n*", "\n\n•")
    # Replace double asterisks (**) with single asterisks (*)
    text = text.replace("**", "*")
    # Place a backslash before specific characters
    characters_to_escape = ["(", ")", "[", "]", "{", "}", "-", ".", "!", "|"]
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