import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        model.eval()
        total_loss, batches = 0.0, 0
        with torch.no_grad():
            for idx, sample in enumerate(test_loader):
                output = model(sample['input'])
                total_loss += loss_function(output, sample['label']).item()
                batches += 1
        loss = total_loss / max(batches, 1)
        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()