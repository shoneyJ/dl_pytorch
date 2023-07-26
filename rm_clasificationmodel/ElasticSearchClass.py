from elasticsearch import Elasticsearch


class ElasticSearchDb:

    def __init__(self, url):
        self.url = url
        self.es = Elasticsearch(url)

    def search(self, _index, doc):
        resp = self.es.search(index=_index,
                              body=doc)
        return resp

    def ingest(self, _index, doc, type):
        self.es.index(
            index=_index,
            body=doc,
            doc_type=type
        )

    def create(self, _index, request_body):

        self.es.indices.create(index=_index, body=request_body)

    def createIndexerPrediction(self, docDict):

        request_body = {
            "settings": {
                "number_of_shards": 5,
                "number_of_replicas": 1
            },
            'mappings': {
                "category_prediction": {

                    'properties': {
                        'id': {'type': 'text'},
                        'name': {'type': 'text'},
                        'category': {'type': 'text'},
                        'predicted': {'type': 'text'},
                        'value': {'type': 'float'},
                    }
                }
            }
        }

        # self.create('english-false-prediction',request_body)

        for val in docDict:

            self.ingest('english-false-prediction', val, 'category_prediction')
    


    def createIndexerProductFeature(self):

        docType="product_feature"
        request_body = {
            "settings": {
                "number_of_shards": 5,
                "number_of_replicas": 1
            },
            'mappings': {
                docType: {

                    'properties': {
                        'id': {'type': 'text'},
                        'name': {'type': 'text'},
                        'shortDesc':{'type': 'text'},
                        'Desc':{'type': 'text'},
                        'category': {'type': 'text'},
                        'catL1': {'type': 'text'},
                        'catL2': {'type': 'text'},
                        'catL3': {'type': 'text'},
                        'catL4': {'type': 'text'},
                        'catL5': {'type': 'text'},
                    }
                }
            }
        }

        self.create('english-product-feature',request_body)

    def ingestProductFeatures(self,docDict):
        for val in docDict:

            self.ingest('english-product-feature', val, 'product_feature')



    def searchProductFeatures(self, _from,_size):
        resp = self.es.search(index="english-product-feature",
                              body={
                                   'from':_from,
                                   'size' :_size,
                              })

        return resp


    def searchDevelopmentProducts(self):
        resp = self.es.search(index="retromotion-indexer_development_products",
                                body={
                                    "_source":
                                    ["_id",
                                    "descriptions",
                                    "descriptionsSource",
                                    "nameSource",
                                    "shortDescriptionSource",
                                    "categoriesSource"
                                    ],
                                    "size":65000

                                })

        return resp
    

    def createIndexerProductFeatureNormal(self):

        docType="product_feature_normal"
        request_body = {
            "settings": {
                "number_of_shards": 5,
                "number_of_replicas": 1
            },
            'mappings': {
                docType: {

                    'properties': {
                       
                        'name': {'type': 'text'},
                        'shortDesc':{'type': 'text'},
                        'Desc':{'type': 'text'},
                        'category': {'type': 'text'},
                        'catL1': {'type': 'text'},
                        'catL2': {'type': 'text'},
                        'catL3': {'type': 'text'},
                        'catL4': {'type': 'text'},
                        'catL5': {'type': 'text'},
                    }
                }
            }
        }

        self.create('english-product-feature-normal',request_body)

    def ingestProductFeaturesNormal(self,docDict):
        for val in docDict:

            self.ingest('english-product-feature-normal', val, 'product_feature_normal')
