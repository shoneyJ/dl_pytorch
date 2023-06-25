from sklearn.feature_extraction.text import CountVectorizer
class Vectorize():
    def __init__(self,ngram_min,ngram_max):
        self.n_gram_vectorizer = CountVectorizer(ngram_range=(ngram_min, ngram_max))

    def ngramFit(self,doc):
        self.n_gram_vectorizer.fit(doc)

    def transform (self,doc):
        self.n_gram_vectorizer.transform(doc)
