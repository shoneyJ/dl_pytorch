
import torch
from TrainClass import Train
from ElasticSearchClass import ElasticSearchDb
from DataFrameClass import DataFrame
from VectorizeClass import Vectorize
from NN import RNN
from evaluate import Evaluate
from predict import *
import pandas as pd



def main():
    es =ElasticSearchDb("http://localhost:9200")

    # resp =es.searchProductFeatures(0,10000)
    # es.createIndexerProductFeature()
    #helper = Helper()
   
    df=DataFrame(es)
    #df.createProductFeature()
    # df_prod= df.setDfProductTaxonomy()
    # doc = df_prod.to_dict(orient='records')

    # es.ingestProductFeatures(doc)
    # dfProductTaxonomyEn=df.getProductTaxonomy(10000,0)    
    df_en = df.getNormal()
    train = Train(df_en)
    train.runBatchWithTLR()
    # predict = Predict()

    # # predict.confusionMatix(df_en)
    # df_prediction=pd.DataFrame(columns=["id",'name','category','predicted','value'])
    # list_row_en = dict (id=None,name=None,category=None,predicted=None,value=None)
    # list =[]
    # for index, row in dfProductTaxonomyEn.iterrows():
    #     predicted =predict.start(row["name"].lower())
    #     actualCategory=helper.normalize(row["category"]) 

    #     if(predicted[0][1]!=actualCategory.lower()):
    #         list_row = dict (id=None,name=None,category=None,predicted=None,value=None)
    #         list_row["id"] =row["id"]
    #         list_row["name"] =row["name"]
    #         list_row["category"] =row["category"]
    #         list_row["predicted"] =predicted[0][1]
    #         list_row["value"] =predicted[0][0]

    #         list.append(list_row)

            

    #         # new_row = pd.Series(list_row)
    #         # df_prediction=pd.concat([df_prediction, new_row.to_frame().T], ignore_index=True)
    # es.createIndexerPrediction(list)


if __name__=="__main__":
    main()