
import torch
from TrainClass import Train
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame
from VectorizeClass import Vectorize
from NN import RNN
from evaluate import Evaluate



def main():
    # es =ElasticSearchDb("http://172.17.212.35:9200")
    es_local =ElasticSearchDb("http://192.168.0.30:9200")
   

    df=DataFrame(es_local)

    df.getNormal()

    df_en = df.getNormal()

    # [df_en, df_de]=df.create()

    # docDict = df_en.to_dict(orient='records')

    # request_body = {
    #         "settings": {
    #             "number_of_shards": 5,
    #             "number_of_replicas": 1
    #         },
    #         'mappings': {
    #             "product_taxonomy": {   
                
    #                 'properties': {
    #                     'name': {'type': 'text'},
    #                     'category': {'type': 'text'},
    #                 }
    #             }
                
    #         }
    #     }

    # es_local.create('english-taxonomy-normal',request_body)
    

    # for val in docDict:
    #     print (val)
    #     es_local.ingest('english-taxonomy-normal',val,'product_taxonomy')




    

    # vectorizer = Vectorize(1,1)
    # vectorizer.setTfidFitTransform(df_en["name"])

    # x =vectorizer.getTfidFitTransform()

    # print(x.shape)

    # vectorizer.fit(doc=df_en["name"])

    # vectorizer.transform(df_en["name"]) 

    # df =df_eng["category"].isnull()

    # bool_name_series = pd.notnull(df_eng["name"])
    # bool_cat_series = pd.isnull(df_eng["name"])
    # print(df_eng[bool_cat_series])
    # inputSize=vectorizer.getTransformedVectorSize()
    # n_hidden = 256*4
    # df_category = df_en.groupby('category')        
    # all_category = list(df_category.groups.keys())        
    # n_category = len(all_category)
    # rnn=RNN(inputSize, n_hidden, n_category)


    # rnn=torch.load('ngram-rnn-classification.pt')
   
    train = Train(df_en)
    train.run(90000)
    train.predict('vaico  centering bush propshaft',2)

    


if __name__=="__main__":
    main()