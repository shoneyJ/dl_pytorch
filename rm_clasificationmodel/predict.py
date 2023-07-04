from NN import *
import sys
from helper import *

class Predict():
    def __init__(self):
         self.rnn = torch.load('ngram-rnn-classification.pt')
         self.helper= Helper()

    def evaluate(self,tensor):
        hidden = self.rnn.initHidden()

            # for i in range(tensor.size()[0]):
            #     output, hidden = self.rnn(tensor[i], hidden)
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

