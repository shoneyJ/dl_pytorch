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
        self.n_hidden = 128
        self.rnn= RNN(self.inputSize, self.n_hidden, self.n_category)
        
        self.lr_low=0.00000000001
        self.lr_max=0.04
        self.learning_rate= 0.00000000000001
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
            p.data.subtract_(p.grad.data, alpha=self.learning_rate)


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

    def runBatch(self):
        batch=[100000]
        total=np.sum(batch)

        learning_rates =[]
        start = time.time()

        for n_iters in batch:
           
            learn_rates=np.linspace(self.lr_low,self.lr_max,total)
            for epoch,lr in zip(range(1, n_iters + 1),learn_rates):
        
                self.learning_rate=lr
                self.lr_low=lr
                category, name, category_tensor, name_tensor = self.data.randomTrainingExample()
                
                output, loss = self.train(category_tensor, name_tensor)
                self.current_loss += loss
                # Print ``iter`` number, loss, name and guess
                if epoch % 1000 == 0:
                    guess, guess_i = self.data.categoryFromOutput(output)
                    correct = '✓' if guess == category else '✗ (%s)' % category
                
                    print('%d %d%% (%s) %.4f %s / %s %s %.10f' % (epoch, 
                                                epoch / n_iters * 100,
                                                self.helper.timeSince(start),
                                                loss,
                                                name, 
                                                guess, 
                                                correct, self.learning_rate))

             
                if epoch % 1000 == 0:
                    if (math.isnan(self.current_loss)!=True):
                        self.all_losses.append(self.current_loss / 1000)
                        learning_rates.append(self.learning_rate)
                        self.current_loss = 0
                    else:
                        break

                total=total-1
            
        # Plot loss curve
        
        plt.figure()
        plt.plot(learning_rates,self.all_losses)
        plt.ylabel("Loss")
        plt.xlabel('Learning rate')        
        # Title and display the plot
        plt.title('Learning Rates vs. Losses')
        plt.savefig("lossinfo")