from tokenizers import Tokenizer
from langchain.vectorstores import Chroma
from semantic_text_splitter import TextSplitter
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.telegram import text_to_docs


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