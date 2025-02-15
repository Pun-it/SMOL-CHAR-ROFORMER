
"""
What does :-
        Encoder DO : Convert 'string' to 'int'
        Decoder DO : Convert 'int' to 'string'
"""

class Tokenizer:

    @staticmethod
    def get_vocab(dataset : str,get_size = False):
        
        vocab = {
            token : index 
            for index,token in enumerate(sorted(list(set(dataset)))) 
        }

        vocab['<unk>'] = len(vocab)

        if get_size:
            return len(vocab.keys())
        
        return vocab

    def __init__(self,vocab):
        

        self.vocab_encode = {str(token):int(index) for token,index in vocab.items()}
        self.vocab_decode = {index:token for token,index in self.vocab_encode.items()}

    def encode(self,text):

        return [self.vocab_encode.get(char,self.vocab_encode["<unk>"]) for char in text]
    
    def decode(self,indexes):

        return "".join([self.vocab_decode.get(idx,"<unk>") for idx in indexes])
    