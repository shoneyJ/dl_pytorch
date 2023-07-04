import torch
import random
import time
import math
import matplotlib.pyplot as plt
from VectorizeClass import Vectorize
class Helper():
    def __init__(self,df_en):
        self.vectorizer = Vectorize(1,1)
        self.df_en=df_en
        self.df_category = self.df_en.groupby('category')
        self.all_category = list(self.df_category.groups.keys())
        self.vectorizer.fit(doc=df_en["name"])
        self.inputSize=self.vectorizer.getVocabLen()
        self.n_categories=len(self.all_category)

    def getIOSize(self):
        return self.inputSize,self.n_categories

    def timeSince(self,since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    def categoryFromOutput(self,output):
        
        _, top_i = output.topk(1)
        
        cat_i = top_i[0].item()
        
        return self.all_category[cat_i], cat_i
    
    def plot(self,data,name):
        plt.figure()
        plt.plot(data)
        plt.savefig(name, dpi=400)

    def nameToTensor(self,name,vectorizer):
        # create empty tensor with 
        inputSize=vectorizer.getVocabLen()
        vectorized=vectorizer.transform(list(name.split()))
        n_vectors=len(vectorized.indices)

        name_tensor=torch.zeros(n_vectors,1, inputSize)
        for i in range(n_vectors):
           
            name_tensor[i][0][vectorized.indices[i]] = 1
        
        return name_tensor
    
    def categoryTensor(self,category):
        return torch.tensor([self.all_category.index(category)], dtype=torch.long)

   
    def randomTrainingExample(self):
     
        randcategory = random.choice(self.all_category)
        # get feature name from the category
        random_feature_indices = self.df_category.indices[randcategory]
        
        index = random_feature_indices[random.randint(0, len(random_feature_indices) - 1)]

        name =self.df_en.iloc[index]["name"]
        
        category_tensor = torch.tensor([self.all_category.index(randcategory)], dtype=torch.long)
        
        name_tensor = self.nameToTensor(name,self.vectorizer)
        
        return randcategory, name, category_tensor, name_tensor