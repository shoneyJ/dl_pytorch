import torch.nn as nn
import torch
import numpy as np
from data import *
import time


from NN import *

class Train():
    def __init__(self,df_en):
        self.df_en=df_en  
        self.data= Data(df_en)
        self.inputSize, self.n_category= self.data.getIOSize()
        self.n_hidden = 128*3
        self.rnn= RNN(self.inputSize, self.n_hidden, self.n_category)
        self.learning_rate=0.005
        self.criterion = nn.NLLLoss()

        self.current_loss = 0
        self.all_losses = []

        self.helper= Helper()
 

    def train(self,category_tensor, name_tensor):
    
        hidden = self.rnn.initHidden()

        self.rnn.zero_grad()

        # for i in range(name_tensor.size()[0]):     

        #     output, hidden = self.rnn(name_tensor[i], hidden)

        output, hidden = self.rnn(name_tensor, hidden)
        loss = self.criterion(output, category_tensor)
        loss.backward()

       
        # Add parameters' gradients to their values, multiplied by learning rate
        for p in self.rnn.parameters():
            p.data.add_(p.grad.data, alpha=-self.learning_rate)



        return output, loss.item()

      
    def run(self,n_iters,print_every,plot_every):
        start = time.time()
        for epoch in range(1, n_iters + 1):
    
            category, name, category_tensor, name_tensor = self.data.randomTrainingExample()
            
            output, loss = self.train(category_tensor, name_tensor)
            self.current_loss += loss
             # Print ``iter`` number, loss, name and guess

            if epoch % print_every == 0:
                guess, guess_i = self.data.categoryFromOutput(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                
                print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, 
                                                epoch / n_iters * 100,
                                                self.helper.timeSince(start),
                                                loss,
                                                name, 
                                                guess, 
                                                correct))

            if epoch % plot_every == 0:
                self.all_losses.append(self.current_loss / plot_every)
                self.current_loss = 0
            
        self.helper.plot(self.all_losses,"loss.png")

        torch.save(self.rnn,'ngram-rnn-classification.pt')
