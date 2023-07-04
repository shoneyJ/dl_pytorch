import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from bs4 import BeautifulSoup
import re
import unicodedata
class DataFrame:
    def __init__(self,es):
        self.es = es
        
    
    def clean(self):

        self.df_eng["name"]	= self.df_eng["name"].str.lower().apply(lambda n:self.normalize(n))        

        self.df_eng["category"]	= self.df_eng["category"].str.lower().apply(lambda c:self.normalize(c))

        self.df_de["name"]	= self.df_de["name"].str.lower().apply(lambda n:self.normalize(n))        

        self.df_de["category"]	= self.df_de["category"].str.lower().apply(lambda c:self.normalize(c))

        self.df_eng=self.df_eng.drop_duplicates()

        self.df_de=self.df_de.drop_duplicates()


        self.df_eng["category"]	= self.df_eng["category"].astype('category')
        self.df_de["category"]	= self.df_de["category"].astype('category')


    
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

   
    def create(self):
        self.df_eng = pd.DataFrame(columns=["id",'name','category'])
        self.df_de = pd.DataFrame(columns=['name','category'])
        resp=self.es.search("retromotion-indexer_development_products",{"_source":["_id","descriptions","descriptionsSource","nameSource","shortDescriptionSource","categoriesSource"],
                           'size' : 65000,
                           "query": {"match_all": {}}})

        for hit in resp['hits']['hits']:
            list_row_en = dict (id=None,name=None,category=None)
            list_row_de = dict (name=None,category=None)
            list_row_en["id"]=hit['_id']
            for name in hit['_source']['nameSource']:
                if (name["language"]=="en"):
                    list_row_en["name"]=name["value"]
                else:
                    list_row_de["name"]=name["value"]
                    

    
            for cats in hit['_source']['categoriesSource']:
                if (cats["language"]=="en"):
                    list_row_en["category"]=cats["label"]
                else:
                    list_row_de["category"]=cats["label"]
            

            if(list_row_en["name"]!=None and list_row_en["category"]!=None):
                new_row = pd.Series(list_row_en)
                self.df_eng=pd.concat([self.df_eng, new_row.to_frame().T], ignore_index=True)

            if(list_row_de["name"]!=None and list_row_de["category"]!=None):
                new_row = pd.Series(list_row_de)
                self.df_de=pd.concat([self.df_de, new_row.to_frame().T], ignore_index=True)
            # df_eng.loc[len(df_eng)] = list_row

        docDict = self.df_eng.to_dict(orient="records")
        self.createIndexerProductCategory(docDict)

        self.clean()

        return [self.df_eng,self.df_de]

    def getNormal(self):
        self.df_eng = pd.DataFrame(columns=['name','category'])
        resp=self.es.search("english-taxonomy-normal",{"_source":["name","category"],
                                                        'size' : 5000,
                                                        "query": {"match_all": {}}})
        for hit in resp['hits']['hits']:
                list_row_en = dict (name=None,category=None)            
                list_row_en["name"]=hit['_source']['name']             
                list_row_en["category"]=hit['_source']['category']

                new_row = pd.Series(list_row_en)
                self.df_eng=pd.concat([self.df_eng, new_row.to_frame().T], ignore_index=True)
        

        
        return self.df_eng

    def getProductTaxonomy(self,_from,_size):

        self.df_taxonomy_all_en = pd.DataFrame(columns=['id','name','category'])
        resp=self.es.search("english-taxonomy-all",{"_source":["id","name","category"],
                                                    'from':_from,
                                                    'size' :_size ,
                                                    "query": {"match_all": {}}})
        for hit in resp['hits']['hits']:
                list_row_en = dict (id=None,name=None,category=None)
                list_row_en["id"]=hit['_source']['id']      
                list_row_en["name"]=hit['_source']['name']             
                list_row_en["category"]=hit['_source']['category']

                new_row = pd.Series(list_row_en)
                self.df_taxonomy_all_en=pd.concat([self.df_taxonomy_all_en, new_row.to_frame().T], ignore_index=True)
        

        
        return self.df_taxonomy_all_en


    def createIndexerTaxonomy(self,docDict):
        request_body = {
        "settings": {
            "number_of_shards": 5,
            "number_of_replicas": 1
        },
        'mappings': {
            "product_taxonomy": {   
            
                'properties': {
                    'name': {'type': 'text'},
                    'category': {'type': 'text'},
                }
            }
            
        }
        }

        self.es.create('english-taxonomy-normal',request_body)
    

        for val in docDict:
            
            self.es.ingest('english-taxonomy-normal',val,'product_taxonomy')

    def createIndexerProductCategory(self,docDict):
        request_body = {
        "settings": {
            "number_of_shards": 5,
            "number_of_replicas": 1
        },
        'mappings': {
            "product_category": {   
            
                'properties': {
                    'id': {'type': 'text'},
                    'name': {'type': 'text'},
                    'category': {'type': 'text'},
                }
            }
        }
        }

        self.es.create('english-taxonomy-all',request_body)
    

        for val in docDict:
            
            self.es.ingest('english-taxonomy-all',val,'product_category')

    


         



