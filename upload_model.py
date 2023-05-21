from huggingface_hub import Repository
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel

print("Starting uploading model")

path_to_folder = "output"
path_to_model = "output/distiluse-base-multilingual-cased-v1-doc2doc"
hf_model_name = "Stern5497/dummy"

model = AutoModel.from_pretrained("Stern5497/dummy")
tokenizer = AutoTokenizer.from_pretrained("Stern5497/dummy")

print("Created model and tokenizer")
