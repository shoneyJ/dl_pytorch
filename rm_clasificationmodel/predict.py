from NN import *
from data import *
import sys

rnn = torch.load('char-rnn-classification.pt')

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def evaluate(self,tensor):
        hidden = self.rnn.initHidden()

        for i in range(tensor.size()[0]):
            output, hidden = self.rnn(tensor[i], hidden)

        return output

def predict(self,name, n_predictions=3):
        print('\n> %s' % name)
        with torch.no_grad():
            output = self.evaluate(self.nameToTensor(name))

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, self.all_category[category_index]))
                predictions.append([value, self.all_category[category_index]])

if __name__ == '__main__':
    predict(sys.argv[1])