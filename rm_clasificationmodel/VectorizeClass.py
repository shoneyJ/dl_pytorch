from sklearn.feature_extraction.text import CountVectorizer
class Vectorize():
    def __init__(self,ngram_min,ngram_max):
        self.vectorizer = CountVectorizer(ngram_range=(ngram_min, ngram_max))

    def ngramFit(self,doc):
        self.vectorizer.fit(doc)

    def transform (self,doc):
        return self.vectorizer.transform(doc)
    
    def getVocabulary(self):
        return self.vectorizer.vocabulary_
    
    