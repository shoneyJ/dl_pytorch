from elasticsearch import Elasticsearch
class ElasticSearchDb:

    def __init__(self,url):
        self.url=url
        self.client = Elasticsearch(url)

    def getFeatures(self):     
        resp = self.client.search(index="retromotion-indexer_development_products",
                     body={"_source":["descriptions","descriptionsSource","nameSource","shortDescriptionSource","categoriesSource"],
                           'size' : 65000,
                           "query": {"match_all": {}}})
        return resp