import torch.nn as nn
import torch
import random
import numpy as np

from NN import RNN

class Train():
    def __init__(self,vectorizer,df_en,all_category,rnn):
        # self.rnn = rnn
        self.learning_rate=0.005
        self.criterion = nn.NLLLoss()
        self.vectorizer =vectorizer
        self.df_en=df_en
        self.all_category=all_category
        self.df_category = self.df_en.groupby('category')
        self.inputSize =  vectorizer.getTransformedVectorSize()     

        self.rnn = rnn
        self.current_loss = 0
        self.all_losses = []
        

    def train(self,category_tensor, name_tensor):
    
        hidden = self.rnn.initHidden()

        self.rnn.zero_grad()

        output, hidden = self.rnn(name_tensor, hidden)

        loss = self.criterion(output, category_tensor)
        loss.backward()

        # for p in self.rnn.parameters():
        #     p.data.add_(-self.learning_rate, p.grad.data)

        return output, loss.item()

    
    def randomTrainingExample(self):
        
        randcategory = random.choice(self.all_category)
        # get feature name from the category
        random_feature_indices = self.df_category.indices[randcategory]
        
        index = random_feature_indices[random.randint(0, len(random_feature_indices) - 1)]

        ngramNameArray = self.vectorizer.getTransformedVectorByIndex(index)

        name =self.df_en.iloc[index]["name"]
        
        category_tensor = torch.tensor([self.all_category.index(randcategory)], dtype=torch.long)
        name_tensor=torch.zeros(1, self.inputSize)
        name_tensor[0][np.where(ngramNameArray==1)[0]] = 1
        
        
        return randcategory, name, category_tensor, name_tensor
    
    
    def categoryFromOutput(self,output):
        
        _, top_i = output.topk(1)
        
        cat_i = top_i[0].item()
        
        return self.all_category[cat_i], cat_i
    

    def run(self,n_iters):
        for epoch in range(1, n_iters + 1):
    
            category, name, category_tensor, name_tensor = self.randomTrainingExample()
            
            output, loss = self.train(category_tensor, name_tensor)
            self.current_loss += loss

            if epoch % 100 == 0:
                guess, guess_i = self.categoryFromOutput(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                
                print('%d %d%% %.4f %s / %s %s' % (epoch, 
                                                epoch / n_iters * 100,
                                                loss,
                                                name, 
                                                guess, 
                                                correct))

            if epoch % 1000 == 0:
                self.all_losses.append(self.current_loss / 1000)
                self.current_loss = 0