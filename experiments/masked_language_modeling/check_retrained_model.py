from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline

#first run retrain.py to creat the model
save_directory = "experiments/masked_language_model/distilbert-base-uncased-finetuned-imdb-accelerate"
# save_directory = "distilbert-base-uncased-finetuned-imdb-accelerate"
model_ = AutoModelForMaskedLM.from_pretrained(save_directory)
tokenizer_ = AutoTokenizer.from_pretrained(save_directory)

mask_filler = pipeline('fill-mask', model = model_, tokenizer = tokenizer_)

text = ["This is a great [MASK].", "Mehdi is a wonderful [MASK]."]
preds = mask_filler(text)

for pred in preds:
    [print(f">>> {pred[i]['sequence']}") for i in range(len(preds))]