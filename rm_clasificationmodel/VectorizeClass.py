from sklearn.feature_extraction.text import CountVectorizer
class Vectorize():
    def __init__(self,ngram_min,ngram_max):
        self.vectorizer = CountVectorizer(ngram_range=(ngram_min, ngram_max))

    def ngramFit(self,doc):
        self.vectorizer.fit(doc)

    def transform (self,doc):
        self.transformedVector = self.vectorizer.transform(doc)
        return self.transformedVector
    
    def getVocabulary(self):
        return self.vectorizer.vocabulary_
    
    def getTransformedVectorByIndex(self,index):
        return self.transformedVector.toarray()[index]
    
    def getTransformedVectorSize(self):
        return self.transformedVector.shape[1]
    
    
    