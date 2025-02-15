import torch
import pandas as pd
from trainer import train
from utils import get_text
from tokenizer import Tokenizer
from dataloader import CharDataset
from torch.utils.data import DataLoader
from blocks import DecoderLayer, DecoderBlock

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

random_text = get_text()

dataset = CharDataset(random_text,seq_length= 128)
char_data = DataLoader(dataset,batch_size = 32)

embedding_dim = 768
seq_len = 32
n_heads = 12          
ff_dim = 3072
n_layers = 20

vocab_size = Tokenizer.get_vocab(random_text,True)

decoder_layer = DecoderLayer(embedding_dim, n_heads, ff_dim).to(DEVICE)
decoder = DecoderBlock(n_layers, decoder_layer,embedding_dim,vocab_size).to(DEVICE)

# Train the decoder block
train(decoder, char_data,vocab_size,epochs = 5)