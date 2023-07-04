import torch
import random
from VectorizeClass import *
from helper import *

class Data():
    def __init__(self,df_en):

        # Define vacobulary

        self.vectorizer = Vectorize(1,1)
        self.df_en=df_en
        self.df_category = self.df_en.groupby('category')
        self.all_category = list(self.df_category.groups.keys())
        self.vectorizer.fit(doc=df_en["name"])
        self.vectorizer.pickleVectorizor()
        self.inputSize=self.vectorizer.getVocabLen()
        self.n_categories=len(self.all_category)

        self.helper= Helper()
        self.helper.dumpAllCategory(self.all_category)

    def getIOSize(self):
        return self.inputSize,self.n_categories

      
    def categoryFromOutput(self,output):
        
        _, top_i = output.topk(1)
        
        cat_i = top_i[0].item()
        
        return self.all_category[cat_i], cat_i
    
    
    def categoryTensor(self,category):
        return torch.tensor([self.all_category.index(category)], dtype=torch.long)
    
    # One-hot vector for category
    def categoryTensor(self,category):
        li = self.all_category.index(category)
        tensor = torch.zeros(1, self.n_categories)
        tensor[0][li] = 1
        return tensor

   
    def randomTrainingExample(self):
     
        randcategory = random.choice(self.all_category)
        # get feature name from the category
        random_feature_indices = self.df_category.indices[randcategory]
        
        index = random_feature_indices[random.randint(0, len(random_feature_indices) - 1)]

        name =self.df_en.iloc[index]["name"]
        
        category_tensor = torch.tensor([self.all_category.index(randcategory)], dtype=torch.long)

        # category_tensor=self.categoryTensor(randcategory)


        
        name_tensor = self.helper.nameToTensor(name)
        
        return randcategory, name, category_tensor, name_tensor
    


    
      