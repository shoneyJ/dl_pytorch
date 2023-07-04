import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
        
    def __init__(self, input_size, hidden_size, output_size):
        
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.h2o = nn.Linear(hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        compbined = torch.cat((input, hidden), 1)
        
        hidden = self.i2h(compbined)
        
        output = self.h2o(hidden)
        output = self.softmax(output)
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    



class Net(nn.Module):
    
    def __init__(self,n_categories, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
            input_combined = torch.cat((category, input, hidden), 1)
            hidden = self.i2h(input_combined)
            output = self.i2o(input_combined)
            output_combined = torch.cat((hidden, output), 1)
            output = self.o2o(output_combined)
            output = self.dropout(output)
            output = self.softmax(output)
            return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)