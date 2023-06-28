class Evaluate():
    def __init__(self,rnn):
        self.rnn=rnn

     # Just return an output given a numpy  array
    def evaluate(self,tensor):
        hidden = self.rnn.initHidden()

        for i in range(tensor.size()[0]):
            output, hidden = self.rnn(tensor[i], hidden)

        return output
    
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
