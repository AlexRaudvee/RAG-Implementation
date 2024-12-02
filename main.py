from transformers import AutoModel, AutoTokenizer

model_name = "sentence-transformers/all-mpnet-base-v2"
save_directory = "./embedding"

# Download and save the model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
