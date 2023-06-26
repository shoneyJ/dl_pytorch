import torch.nn as nn
class Train():
    def __init__(self,rnn,lr):
        self.rnn = rnn
        self.lr=lr
        self.criterion = nn.NLLLoss()


    def train(self,category_tensor, name_tensor):
    
        hidden = self.rnn.initHidden()

        self.rnn.zero_grad()

        for i in range(name_tensor.size()[0]):
            output, hidden = self.rnn(name_tensor[i], hidden)

        loss = self.criterion(output, category_tensor)
        loss.backward()

        for p in self.rnn.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)

        return output, loss.item()