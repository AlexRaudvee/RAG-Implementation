import google.generativeai as genai

from envvar import GEMINI_API_KEY
from transformers import BertTokenizer, BertModel

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name = "gemini-1.5-flash-002", system_instruction="""When you answer the questions with the provided context, 
                              do not say that you rely on context, just answer the question, like pretending that you knew this information before it was provided to you. 
                              Make sure that your answer is always structured and fully answers all aspects of the question from user, make use of all information that you have! 
                              Keep in mind that user can do typos, so if there is a typo just guess what is the word is!""")

# Load pre-trained BERT model and tokenizer for computing text similarity
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
similarity_model = BertModel.from_pretrained('bert-base-uncased')