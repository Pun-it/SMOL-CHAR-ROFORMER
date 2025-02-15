import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def generator(model,tokenizer,seed_text='',limit = 100):
    model.eval()
    with torch.no_grad():
        
        tokens = tokenizer.encode(seed_text)
        
        tokens = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
        # all_tokens = tokens
        for _ in range(limit):

            tokens = tokens.type(torch.LongTensor).to(DEVICE)
            output = model(tokens)
            
            probabilities = F.softmax(output[0, -1], dim=0)
            next_token = torch.multinomial(probabilities, 1).item()
            tokens = torch.cat([tokens,torch.Tensor([[next_token]]).to(DEVICE)], dim = 1)

            # tokens = all_tokens[0][1:].unsqueeze(0).to(DEVICE)
        generated_text = ' '.join(tokenizer.decode(token) for token in tokens.cpu().numpy())
        
        return generated_text