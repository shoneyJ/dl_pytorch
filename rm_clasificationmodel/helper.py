import pickle
from bs4 import BeautifulSoup
import re
import unicodedata
import time
import math
import matplotlib.pyplot as plt
import torch
class Helper():
    def __init__(self):
        pass
    def loadVectorizer(self):
        return pickle.load(open("vector.pickel", "rb"))
    
    def dumpAllCategory(self,allCategory):
        pickle.dump(allCategory, open("category.pickel", "wb"))

    def loadAllCategory(self):
        return pickle.load(open("category.pickel", "rb"))
    
    def normalize(self,doc):

        # Remove html tags 
        soup = BeautifulSoup(doc, 'html.parser')
        text =soup.get_text()
        text = (re.sub('[-\W]+', ' ', text))
        text = (re.sub('(?<=\d) (?=\d)', '', text))
        text = (re.sub("([a-z]\d+)|(\d+)", '', text))
        
        
        return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn')
        # text= text.replace('-', '')
    
    def timeSince(self,since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    def plot(self,data,name):
        plt.figure()
        plt.plot(data)
        plt.savefig(name, dpi=400)

    
    def nameToTensor(self,name):
        vectorizer=self.loadVectorizer()
        inputSize=len(vectorizer.vocabulary_)
        vectorized=vectorizer.transform(list(name.split()))
        # n_vectors=len(vectorized.indices)

        # name_tensor=torch.zeros(n_vectors,1, inputSize)
        # for i in range(n_vectors):
           
        #     name_tensor[i][0][vectorized.indices[i]] = 1
        
        # return name_tensor

        name_tensor=torch.zeros(1, inputSize)

        for index in vectorized.indices:
            name_tensor[0][index] = 1
        
        return name_tensor

    
    def getCategoryByIndex(self,category_index):
        all_category = self.loadAllCategory()
        return all_category[category_index]