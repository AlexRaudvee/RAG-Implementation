import os
import torch
import requests

from tqdm import tqdm
from bs4 import BeautifulSoup
from googlesearch import search
from tokenizers import Tokenizer
from langchain.vectorstores import Chroma
from semantic_text_splitter import TextSplitter
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders.telegram import text_to_docs

from config import tokenizer, similarity_model


def upload_document(file_path):
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


def initialize_vector_db(docs):
    print("EMBEDDING THE DOCUMENTS: ...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./db")
    vector_db.persist()
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
    
    
def get_google_links(query, num_results=5):
    # Perform the search
    results = search(query, num_results=num_results)
    # Extract the URLs
    links = [result for result in results]
    return links
        
    
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
        links = [a['href'] for a in soup.find_all('a', href=True) if "https://" in a['href']]

        # Save text to a file
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(text)

        # Get the absolute file path
        file_path = os.path.abspath(output_filename)

        # Return the file path and the list of links
        return file_path, links
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, []
    
    
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
    
    
def search_info_to_txt(question: str, top: int = 5, similarity: float = 0.7):

    os.makedirs("web_info", exist_ok=True)
    
    links_main = get_google_links(question, top)

    all_links_child = []
    
    parent_num = 1
    child_num= 1
    for link_root in tqdm(links_main, desc="Processing Root Links", dynamic_ncols=True):
        file_path, links_child = parse_and_save(link_root, f"web_info/root_{parent_num}.txt")
        links_child = get_relevant_links(question, links_child, similarity)
        all_links_child = all_links_child + links_child
        parent_num += 1
        for link_child in links_child:
            try:
                file_path, _ = parse_and_save(link_child, f"web_info/number_{child_num}.txt")
                child_num += 1
            except:
                continue
    
    links_visited = links_main + all_links_child
    
    return links_visited