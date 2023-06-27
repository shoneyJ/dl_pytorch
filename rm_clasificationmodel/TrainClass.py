import torch.nn as nn
import torch
import random

from NN import RNN

class Train():
    def __init__(self,countVectorizer,df_en):
        # self.rnn = rnn
        self.learning_rate=0.005
        self.criterion = nn.NLLLoss()
        self.countVectorizer =countVectorizer
        self.df_en=df_en
        self.df_category = self.df_en.groupby('category')        
        self.all_category = list(self.df_category.groups.keys())

        self.n_hidden = 256
        self.n_category = len(self.all_category)
        self.inputSize=self.countVectorizer.getTransformedVectorSize()

        self.rnn = RNN(self.inputSize, self.n_hidden, self.n_category)


    def train(self,category_tensor, name_tensor):
    
        hidden = self.rnn.initHidden()

        self.rnn.zero_grad()

        output, hidden = self.rnn(name_tensor, hidden)

        loss = self.criterion(output, category_tensor)
        loss.backward()

        for p in self.rnn.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)

        return output, loss.item()

    
    def randomTrainingExample(self):
        
        randcategory = random.choice(self.all_category)
        # get feature name from the category
        random_feature_indices = self.df_category.indices[randcategory]
        
        index = random_feature_indices[random.randint(0, len(random_feature_indices) - 1)]

        ngramNameArray = self.countVectorizer.getTransformedVectorByIndex(index)

        name =self.df_en.iloc[index]["name"]
        
        category_tensor = torch.tensor([self.all_category.index(randcategory)], dtype=torch.long)
        name_tensor = torch.tensor(ngramNameArray)
        
        return randcategory, name, category_tensor, name_tensor
    
    
    def categoryFromOutput(self,output):
        
        _, top_i = output.topk(1)
        
        cat_i = top_i[0].item()
        
        return self.all_category[cat_i], cat_i
    

    def run(self,n_iters):
        for epoch in range(1, n_iters + 1):
    
            category, name, category_tensor, name_tensor = self.randomTrainingExample()
            
            output, loss = self.train(category_tensor, name_tensor)
            current_loss += loss

            if epoch % 5000 == 0:
                guess, guess_i = self.categoryFromOutput(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                
                print('%d %d%% %.4f %s / %s %s' % (epoch, 
                                                epoch / n_iters * 100,
                                                loss,
                                                name, 
                                                guess, 
                                                correct))

            if epoch % 1000 == 0:
                self.all_losses.append(current_loss / 1000)
                current_loss = 0