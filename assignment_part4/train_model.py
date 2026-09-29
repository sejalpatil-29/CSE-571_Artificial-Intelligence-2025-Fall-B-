from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)
    torch.save(model.state_dict(), 'saved/saved_model.pkl')


    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            optimizer.zero_grad()
            output = model(sample['input'])
            loss = loss_function(output, sample['label'])
            loss.backward()
            optimizer.step()

        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        losses.append(test_loss)
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), 'saved/saved_model.pkl')
        print(f'Epoch {epoch_i+1}/{no_epochs}  test loss: {test_loss:.4f}')

    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('test loss')
    plt.savefig('saved/loss.png')
    plt.show()



if __name__ == '__main__':
    no_epochs = 100
    train_model(no_epochs)