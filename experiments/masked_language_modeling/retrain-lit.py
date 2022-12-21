import matplotlib
import torch
import math
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import collections
import numpy as np
from transformers import default_data_collator
from transformers import TrainingArguments 
from transformers import Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm


'''
******** defining model and tokenizer ******************
'''
#importing the model 
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
print("number of model parameters: ", round(model.num_parameters() / 1_000_000))

#tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

'''
******** domain adaptation ******************
'''

'''
loading the data 
'''
imdb_dataset = load_dataset('imdb')

'''
preprocessing data for masked language modeling task
'''
def tokenizer_fn(data):
    data_text = data['text']
    data_tokenized = tokenizer(data_text)
    return data_tokenized

tokenized_datasets = imdb_dataset.map(tokenizer_fn, batched=True, remove_columns=["text", "label"])
print(tokenized_datasets)
# group all data together and split the result into chunks.
print(tokenizer.model_max_length) # for this model = 512

'''
preparing lm (language modeling) dataset
'''
chunk_size = 128
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()  #!!!!!!!!!!
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

'''
defining data collator
'''
#Note: A data collator is just a function that takes a list of samples and converts them into a batch
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

'''
splitting and down sampling the data to reduce the traning time 
'''
train_size = 10_000
valid_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=valid_size, seed=42
)

print(downsampled_dataset)

batch_size = 64

train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

eval_dataloader = DataLoader(
    downsampled_dataset["test"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

'''
Training the model (GPU required)
'''
#loading refresh refresh vesion of the pretrain model
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint) 
optimizer = AdamW(model.parameters(), lr = 5e-5)
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
) 

# specifying the learning rate scheduler
num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader) # number of batches
num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
output_dir = "experiments/masked_language_modeling/" + model_name
progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    model.train()
    for bach in train_dataloader:
        outputs = model(**bach)
        loss = outputs.loss 
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        
        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(downsampled_dataset["test"])]   # len eval dataset
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)




