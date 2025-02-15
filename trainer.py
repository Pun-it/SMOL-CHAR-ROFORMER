import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

def train(model, dataset, vocab_size, epochs=1):

    print(f'Training for {epochs} EPOCHS')

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    model.train()

    for epoch in tqdm(range(epochs)):
        total_loss = 0

        for enum,batch in tqdm(enumerate(dataset)):
            # Unpack batch
            x, y = batch
            x,y = x.to(DEVICE),y.to(DEVICE)
            # Zero the gradient
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x)

            # Compute loss
            loss = F.cross_entropy(y_pred.view(-1,vocab_size),y.view(-1))
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            print(f'Loss : {loss.item()}')
        # print(total_loss)
        
        
        torch.save(model.state_dict(),f'saved_model{epoch}.pth')
        
        # Print epoch loss
        avg_loss = total_loss / len(dataset)
        print(f'Epoch {epoch + 1}, Epoch Loss: {avg_loss:.4f}')
