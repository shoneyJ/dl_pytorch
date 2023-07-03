from elasticsearch import Elasticsearch
class ElasticSearchDb:

    def __init__(self,url):
        self.url=url
        self.es = Elasticsearch(url)

    def search(self,_index,doc):     
        resp = self.es.search(index=_index,
                     body=doc)
        return resp
    
    def ingest(self,_index,doc,type):
        self.es.index(
            index=_index,
            body=doc,
            doc_type=type
            )
        
    def create(self,_index,request_body):

        self.es.indices.create(index=_index, body=request_body)
