from src.transformer import CTransformer
import torch
from torch import nn 
import torch.nn.functional as F
from torchtext.datasets import IMDB, AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext import vocab
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

tbw = SummaryWriter(log_dir = './runs') # Tensorboard logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("hi i'm testing packaging")
exit()

'''
===============================================================================================
 load data
 This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative)
===============================================================================================
'''
# This is the page I used to load data with torchtext
# !!! https://torchtutorialstaging.z5.web.core.windows.net/beginner/text_sentiment_ngrams_tutorial.html !!!

train_iter, test_iter = AG_NEWS()

# print(len(list(train_iter)))
# print(len(list(test_iter)))

# print(next(iter(train_iter)))
# print(next(iter(test_iter)))

# for label, line in train_iter: 
#     print(f"Label: {label}")
#     print(f"Line: '{line}'")
#     break

tokenizer = get_tokenizer('basic_english')
counter = Counter()
seq_length = []
for (label, line) in train_iter:
    seq = tokenizer(line)
    counter.update(seq)
    seq_length.append(len(seq))

for (label, line) in test_iter:
    seq = tokenizer(line)
    counter.update(seq)
    seq_length.append(len(seq))

vocab_dic = vocab(counter, min_freq=1, specials=['<pad>'])

'''
===============================================================================================
 HYPERPARAMETERS
===============================================================================================
'''
learning_rate = 0.0001
lr_warmup = 10_000
batch_size = 4  
emsize = 128   # embedding size 
num_heads = 8  # num of transformer head 
depth = 6      # no of transformer blocks
num_epochs = 1 
useValidateSet = True
num_class = len(set([label for (label, text) in train_iter]))
max_seq_lenght = max(seq_length) # this is important for position embedding
vocab_size = len(vocab_dic) * 2 # "this is important for token embedding" 
vocab_size = 500_000

default_index = 0
vocab_dic.set_default_index(default_index)

'''
===============================================================================================
 Data preprocessing
===============================================================================================
'''
text2ids_transform = lambda x: [vocab_dic[token] for token in tokenizer(x)]
label_transform = lambda x: int(x) - 1

# print(text2token_transform('here is the an example'))

def collate_batch(batch): 
    label_list, text_list = [], [] 
  
    for (_label, _text) in batch: 
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text2ids_transform(_text), dtype=torch.int64)
        text_list.append(processed_text)
    
    label_list = torch.tensor(label_list, dtype=torch.int64)

    padded_value = vocab_dic['<pad>']
    max_size = max([item.size(0) for item in text_list]) 

    padded = [torch.cat([item,torch.tensor([padded_value]).expand(
         max_size- len(item))]) for item in text_list]

    text_list = torch.cat([item[None] for item in padded])

    return label_list.to(device), text_list.to(device) 

train_loader = DataLoader(list(train_iter), 
                               batch_size=batch_size, 
                               shuffle=True, 
                               collate_fn=collate_batch)

test_loader = DataLoader(list(test_iter), 
                               batch_size=batch_size, 
                               shuffle=True, 
                               collate_fn=collate_batch)

# for i, batch in enumerate(train_loader):
#     print(i, batch)
#     break

'''
===============================================================================================
 Splitting the train data into train and validate (optional)
===============================================================================================
'''
if useValidateSet : 
    train_dataset = list(train_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_loader = DataLoader(split_train_, batch_size = batch_size,
                                shuffle = True, collate_fn = collate_batch)
    valid_loader = DataLoader(split_valid_, batch_size = batch_size,
                                shuffle = True, collate_fn = collate_batch)
else: 
    valid_loader = test_loader  
'''
===============================================================================================
 train and evaluate functions
===============================================================================================
'''
model = CTransformer(k = emsize, heads = num_heads, depth = depth, max_seq_length = max_seq_lenght, 
                        vocab_size = vocab_size, num_classes = num_class).to(device)

opt = torch.optim.Adam(lr = learning_rate, params = model.parameters())
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / batch_size), 1.0))
criterion =  torch.nn.NLLLoss()

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        opt.zero_grad()
        predited_label = model(text)
        loss = criterion(predited_label, label)
        loss.backward()
        # clip gradients
        # - If the total gradient vector has a length > 1, we clip it back down to 1.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            
            tbw.add_scalar('classification/train-loss', epoch, total_acc/total_count)

            total_acc, total_count = 0, 0
            start_time = time.time()
            
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predited_label = model(text)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

'''
===============================================================================================
 Start training
===============================================================================================
'''
for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    train(train_loader)
    accu_val = evaluate(valid_loader)

    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
                 
'''
#===============================================================================================
# test the model
#===============================================================================================
'''
print('Checking the results of test dataset.')
accu_test = evaluate(test_loader)
print(f'test accuracy {accu_test: 8.3f}')

'''
#===============================================================================================
# test on random news
#===============================================================================================
'''
ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor([text_pipeline(text)])
        output = model(text)
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. - Four days ago, Jon Rahm was \
    enduring the season's worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday's first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he'd never played the \
    front nine at TPC Southwind."

model = model.to(device)

print("This is a %s news" %ag_news_label[predict(ex_text_str, text2ids_transform)])