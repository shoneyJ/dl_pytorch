from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
class Vectorize():
    def __init__(self,ngram_min,ngram_max):
        self.vectorizer = CountVectorizer(ngram_range=(ngram_min, ngram_max))
        self.TfidV = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')

    def fit(self,doc):
        self.vectorizer.fit(doc)
    
    def getVocabLen(self):
        return len(self.vectorizer.vocabulary_)

    def transform (self,doc):
        self.transformedVector=self.vectorizer.transform(doc)
        return self.transformedVector

    def getVocabulary(self):
        return self.vectorizer.vocabulary_
    
    def getTransformedVectorByIndex(self,index):
        return self.transformedVector.toarray()[index]
    
    def getTransformedVectorSize(self):
        return self.transformedVector.shape[1]


    def setTfidFitTransform(self,doc):
        self.TfidVectorized=self.TfidV.fit_transform(doc) 

    def getTfidFitTransform(self):
       return  self.TfidVectorized       
    
    