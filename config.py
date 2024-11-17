import google.generativeai as genai
from envvar import GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(model_name = "gemini-1.5-flash-002", system_instruction="When you answer the questions with the provided context, do not say that you rely on context, just answer the question, like pretending that knew this information before it was provided to you.")
