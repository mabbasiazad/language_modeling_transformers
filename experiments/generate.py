from src.transformer import GTransformer
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
import random, tqdm, sys, math, gzip
from torch.utils.data import DataLoader, Dataset
import torch.distributions as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# prepare enwik8 data

SEQ_LEN = 256
batch_size = 32
vocab_size = 256
# NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
# power of two.
lr = 0.0001
lr_warmup = 5000

'''
============================================================================================
Utility Functions
===========================================================================================
'''
#The model is a language model that operates on characters. 
#Therefore, this model does not need a tokenizer. 
#The following function can instead be used for encoding and decoding:

def encode(list_of_strings, pad_token_id=0):
    max_length = max([len(string) for string in list_of_strings])

    # create emtpy tensors
    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        if not isinstance(string, bytes):
            string = str.encode(string) # get the ASCI code

        input_ids[idx, :len(string)] = torch.tensor([x  for x in string])
        attention_masks[idx, :len(string)] = 1

    return input_ids, attention_masks

def decode(outputs_ids):
    decoded_outputs = []
    for output_id in outputs_ids:
        decoded_outputs.append("".join([chr(x) for x in output_id]))
    return decoded_outputs

# This blog post shows how temperature sampling works
# https://lukesalamone.github.io/posts/what-is-temperature/#:~:text=Temperature%20is%20a%20parameter%20used,to%20play%20with%20it%20yourself.
def temp_sampling(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

def sample_batch(data, seq_length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.
    For each input instance, it also slices out the sequence that is shofted one position to the right, to provide as a
    target for the model.
    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - seq_length - 1)

    # Slice out the input sequences
    seqs_inputs  = [data[start:start + seq_length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + seq_length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target

def generate_sequence(model, seed, seq_length, generate_length=600, temperature=0.5, verbose=False):
    """
    Sequentially samples a sequence from the model, token by token.
    :param model:
    :param seed: The sequence to start with.
    :pararm seq_length: other names token_nunber / max_seq_length
    :param generate length: The total number of characters to be generated.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.
    :return: The sampled sequence, including the seed.
    """

    sequence = seed.detach().clone()

    if verbose: # Print the seed, surrounded by square brackets
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(generate_length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-seq_length:]  #!!! this is the important trick

        # Run the current input through the model 
        # we treat the data as a single batch entery to the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = temp_sampling(output[0, -1, :], temperature) #!!! sampling form the last output seq 

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0) # !!! Append the sampled token to the sequence

    print()
    return seed

'''
===============================================================================================
Data load and process
===============================================================================================
'''
def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.frombuffer(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

#note: change the current path to exprement folder
path = './data/enwik8.gz'
data_train, data_val, data_test = enwik8(path)
data_train = torch.cat([data_train, data_val], dim = 0)

# print(data_train[0:15])
# print(decode(data_train[0:1000]))

'''
============================================================================================
Machine learning modeling
=============================================================================================
'''
model = GTransformer(k=256, heads=8, depth=12, max_seq_length=SEQ_LEN, vocab_size= vocab_size)
opt = torch.optim.Adam(lr=lr, params=model.parameters())
# Linear learning rate warmup
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / batch_size), 1.0))
criterion = torch.nn.NLLLoss()

# !!! Training loop !!!
# -- We don't loop over the data, instead we sample a batch of random subsequences each time.

isinstances_seen = 0
num_iter = 1_000_000 #very large value so you can keep running until the output looks good."
test_every = 5

for i in tqdm.trange(num_iter):  
    input, target = sample_batch(data_train, seq_length = SEQ_LEN, batch_size = batch_size)
    isinstances_seen += input.size(0)
    output = model(input)
    # print(target, target.size())
    # print(output, output.size())
    loss = criterion(output.transpose(1, 2), target)
    # print(loss)
    loss.backward()
    # - If the total gradient vector has a length > 1, we clip it back down to 1.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.zero_grad()
    opt.step()

    if i % test_every == 0 :
        with torch.no_grad():

            ## Sample and print a random sequence
            # Slice a random seed from the test data, and sample a continuation from the model.
            seedfr = random.randint(0, data_test.size(0) - SEQ_LEN)
            seed = data_test[seedfr:seedfr + SEQ_LEN].to(torch.long)

            if torch.cuda.is_available():
                seed = seed.cuda()

            generate_sequence(model, seed=seed, seq_length=SEQ_LEN, verbose=True, generate_length=600)




