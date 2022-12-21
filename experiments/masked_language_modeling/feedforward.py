import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

#importing the model 
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

print("number of model parameters: ", round(model.num_parameters() / 1_000_000))

#tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

'''
******** test the model ******************
'''
text = ["This is a great [MASK]."]

print("mask token id")
print(tokenizer.mask_token_id)


inputs =  tokenizer(text, padding=True ,return_tensors="pt")
print("input ids ")
print(inputs)
outputs = model(**inputs)
print("ouput of the model")
print(outputs.logits.size())

mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
print('mask token index:', mask_token_index)
mask_token_logits = outputs.logits[0, mask_token_index, :]
print(mask_token_logits.size())

top_5_unmask_ids = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
print(top_5_unmask_ids)
top_5_unmask_token = tokenizer.decode(top_5_unmask_ids)
print(top_5_unmask_token)

