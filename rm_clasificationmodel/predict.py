from matplotlib import ticker
from NN import *
import sys
from helper import *
from data import *

class Predict():
    def __init__(self):
         self.rnn = torch.load('ngram-rnn-classification.pt')
         self.helper= Helper()

    def evaluate(self,tensor):
        hidden = self.rnn.initHidden()
        output, hidden = self.rnn(tensor, hidden)
        return output

    def start(self,name, n_predictions=1):
           
            with torch.no_grad():
                output = self.evaluate(self.helper.nameToTensor(name))

                # Get top N categories
                topv, topi = output.topk(n_predictions, 1, True)
                predictions = []

                for i in range(n_predictions):
                    value = topv[0][i].item()
                    category_index = topi[0][i].item()
                    # print('(%.2f) %s' % (value, self.helper.getCategoryByIndex(category_index)))
                    predictions.append([value, self.helper.getCategoryByIndex(category_index)])
            
            return predictions
    
    def confusionMatix(self,df_en):
        # Keep track of correct guesses in a confusion matrix
        data = Data(df_en)
        n_categories = len(data.all_category)

        batch = random.choices(data.all_category,k=20)
        
        confusion = torch.zeros(n_categories, n_categories)
        n_confusion = 10000

        # Go through a bunch of examples and record which are correctly guessed
        for i in range(n_confusion):
            category, name, category_tensor, name_tensor = data.randomTrainingExample()
            if category in batch:
                output = self.evaluate(name_tensor)
                guess, guess_i = data.categoryFromOutput(output)
                category_i =data.all_category.index(category)
                confusion[category_i][guess_i] += 1
        
        # Normalize by dividing every row by its sum
        for i in range(20):
            confusion[i] = confusion[i] / confusion[i].sum()
       
        
        # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion.numpy())
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + data.all_category, rotation=45)
        ax.set_yticklabels([''] + data.all_category)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # sphinx_gallery_thumbnail_number = 2
        plt.show()
        plt.savefig('confusion.png', dpi=400)
    



