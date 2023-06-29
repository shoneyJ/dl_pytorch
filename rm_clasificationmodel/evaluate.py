from matplotlib import pyplot as plt, ticker
from TrainClass import Train
import torch


class Evaluate():
    def __init__(self,all_categories):
       
        self.all_categories = all_categories
        self.rnn = torch.load('ngram-rnn-classification.pt')

     # Just return an output given a numpy  array
    def evaluate(self,tensor):
        hidden = self.rnn.initHidden()

        for i in range(tensor.size()[0]):
            output, hidden = self.rnn(tensor[i], hidden)

        return output
    
    def confusionMatix(self):
        # Keep track of correct guesses in a confusion matrix
        n_categories = len(self.all_categories)
        confusion = torch.zeros(n_categories, n_categories)
        n_confusion = 10000

        # Go through a bunch of examples and record which are correctly guessed
        for i in range(n_confusion):
            category, name, category_tensor, name_tensor = Train.randomTrainingExample()
            output = self.evaluate(name_tensor)
            guess, guess_i = self.categoryFromOutput(output)
            category_i = self.all_categories.index(category)
            confusion[category_i][guess_i] += 1
        
        # Normalize by dividing every row by its sum
        for i in range(len):
            confusion[i] = confusion[i] / confusion[i].sum()
        
        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion.numpy())
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + self.all_categories, rotation=90)
        ax.set_yticklabels([''] + self.all_categories)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # sphinx_gallery_thumbnail_number = 2
        plt.show()

    
    # def predict(self,line, n_predictions=3):
    #     output = self.evaluate((lineToTensor(line)))

    #     # Get top N categories
    #     topv, topi = output.data.topk(n_predictions, 1, True)
    #     predictions = []

    #     for i in range(n_predictions):
    #         value = topv[0][i]
    #         category_index = topi[0][i]
    #         print('(%.2f) %s' % (value, self.all_categories[category_index]))
    #         predictions.append([value, self.all_categories[category_index]])

    #     return predictions
