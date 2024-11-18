import google.generativeai as genai

from envvar import GEMINI_API_KEY
from transformers import BertTokenizer, BertModel

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name = "gemini-1.5-flash-002", system_instruction="When you answer the questions with the provided context, do not say that you rely on context, just answer the question, like pretending that knew this information before it was provided to you.")

# Load pre-trained BERT model and tokenizer for computing text similarity
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
similarity_model = BertModel.from_pretrained('bert-base-uncased')