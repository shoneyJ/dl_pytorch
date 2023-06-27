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

        for i in range(name_tensor.size()[0]):
            output, hidden = self.rnn(name_tensor[i], hidden)

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