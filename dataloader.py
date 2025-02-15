from torch.utils.data import Dataset
from tokenizer import Tokenizer
import torch

class CharDataset(Dataset):

    def __init__(self,text,seq_length):

        
        vocab = Tokenizer.get_vocab(text)
        tokenizer = Tokenizer(vocab)
        self.text = text
        self.seq_length = seq_length
        self.encoded_text = tokenizer.encode(text)

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):

        input_seq = self.encoded_text[idx: idx + self.seq_length]
        target_seq = self.encoded_text[idx + 1: idx + self.seq_length + 1]

        return (torch.tensor(input_seq), torch.tensor(target_seq))
    
