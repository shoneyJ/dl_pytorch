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

        self.lr_low=0.0075
        self.lr_max=0.0125
        self.learning_rate= 0.00000000000001
        self.criterion = nn.NLLLoss()

        self.current_loss = 0
        self.all_losses = []

        self.helper= Helper()

        # Define the number of iterations or epochs for the up and down phases
        self.up_iterations = 1000
        self.down_iterations = 1000


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

            learn_rates=np.linspace(self.lr_low,self.lr_max,n_iters)
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



    def runBatchWithTLR(self):
        batch=[10000,10000,10000,10000,10000,10000,10000,10000,10000,10000]
        start = time.time()
        no_batch=0
        no_iterations=0
        learning_rates =[]
        times =[]

        for n_iters in batch:

            no_batch =no_batch+1
            isEven=no_batch % 2
            learn_rates=np.linspace(self.lr_low,self.lr_max,n_iters)
            if isEven==0:
                learn_rates=np.flip(learn_rates)

            for epoch,lr in zip(range(1, n_iters + 1),learn_rates):
                no_iterations= no_iterations+1

                self.learning_rate=lr
                category, name, category_tensor, name_tensor = self.data.randomTrainingExample()

                output, loss = self.train(category_tensor, name_tensor)
                self.current_loss += loss
                # Print ``iter`` number, loss, name and guess
                if epoch % 1000 == 0:
                    guess, guess_i = self.data.categoryFromOutput(output)
                    correct = '✓' if guess == category else '✗ (%s)' % category

                    print('%d %d%% (%s) %.4f %s / %s %s %.10f' % (no_iterations,
                                                no_iterations / np.sum(batch) * 100,
                                                self.helper.timeSince(start),
                                                loss,
                                                name,
                                                guess,
                                                correct, self.learning_rate))


                if epoch % 5000 == 0:
                    if (math.isnan(self.current_loss)!=True):
                        self.all_losses.append(self.current_loss / 1000)
                        learning_rates.append(self.learning_rate)
                        times.append(self.helper.secondsSince(start))
                        self.current_loss = 0
                    else:
                        break

        # Plot loss curve
        # Initialise the subplot function using number of rows and columns
        # figure, axs = plt.subplots(2,sharex=True)
        # figure.suptitle("Loss vs Learning rate and time with TLR")
        # # plt.figure()
        # axs[0].plot(self.all_losses,learning_rates,'tab:orange')
        # axs[1].plot(self.all_losses,times,'tab:green')

        # for ax in axs.flat:
        #     ax.set(xlabel='Loss')

        # plt.ylabel("Loss")
        # plt.xlabel('Learning rate')
        # # # Title and display the plot
        # ax1.set_title('Losses Vs Learning rate')
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)

        plt.plot(learning_rates, self.all_losses, marker='o', linestyle='-')
        plt.title('Learning Rate vs. Loss')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        # plt.xscale('log')  # Log scale for learning rate for better visualization
        plt.grid(True)

        # Create a plot for time vs. loss
        plt.subplot(1, 2, 2)
        plt.plot(times, self.all_losses, marker='o', linestyle='-')
        plt.title('Training Time vs. Loss')
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.tight_layout()

        plt.savefig("lossinfo")